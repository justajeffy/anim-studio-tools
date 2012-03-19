/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: http://svn/drd/apps/bee/trunk/lib/gl/Shader.cpp $"
 * SVN_META_ID = "$Id: Shader.cpp 45023 2010-09-07 23:33:22Z stephane.bertout $"
 */

#include "Shader.h"

#include <drdDebug/log.h>
#include <stdio.h>
#include <iostream>
#include <boost/assert.hpp>

using namespace bee;
using namespace drd;
DRD_MKLOGGER(L,"drd.bee.gl.Shader");

Shader::Shader( GLenum shaderType )
: m_Type( shaderType )
, m_Code( NULL )
{
	m_ID = glCreateShader( m_Type );
}

Shader::~Shader()
{
	deleteCode();
	glDeleteShader( m_ID );
}

void Shader::deleteCode()
{
	if ( m_Code )
	{
		delete[] m_Code;
		m_Code = NULL;
	}
}

bool Shader::load( String a_FileName )
{
	m_FileName = a_FileName;
	m_Code = fileRead( a_FileName );
	if ( m_Code ) return true;
	else return false;
}

bool Shader::compile( const char * a_IncludeCode )
{
	if ( m_Code == NULL ) return false;

	if ( a_IncludeCode == NULL )
	{
		const char * code = m_Code;
		glShaderSource( m_ID, 1, &code, NULL );
	}
	else
	{
		const char * codes[ 2 ] =
		{ a_IncludeCode, m_Code };
		glShaderSource( m_ID, 2, codes, NULL );
	}

	glCompileShader( m_ID );

	int param;
	glGetShaderiv( m_ID, GL_COMPILE_STATUS, &param );
	if ( param )
	{
		return true;
	}
	else
	{
		String sInfoLog;
		getInfoLog( sInfoLog );

		std::cout << "Shader " << m_FileName.c_str() << " does not compile ! => " << sInfoLog.c_str() << std::endl;

		return false;
	}
}

char* Shader::fileRead( String a_FileName )
{
	FILE *fp;
	char *content = NULL;

	int count = 0;

	BOOST_ASSERT( a_FileName != "" );

	fp = fopen( a_FileName.c_str(), "rt" );
	if ( fp != NULL )
	{
		fseek( fp, 0, SEEK_END );
		count = ftell( fp );
		rewind( fp );
		if ( count > 0 )
		{
			content = new char[ count + 1 ];
			count = fread( content, sizeof(char), count, fp );
			content[ count ] = '\0';
		}
		fclose( fp );
	}
	else
	{
		DRD_LOG_ERROR( L, "File does not exist or can't open: "<< a_FileName );
	}

	return content;
}

void Shader::getInfoLog( String & o_InfoLog )
{
	int infologLength = 0;
	int charsWritten = 0;
	char *infoLog;

	glGetShaderiv( m_ID, GL_INFO_LOG_LENGTH, &infologLength );

	if ( infologLength > 0 )
	{
		infoLog = new char[ infologLength ];
		glGetShaderInfoLog( m_ID, infologLength, &charsWritten, infoLog );
		o_InfoLog = String( infoLog );
		delete[] infoLog;
	}
}


/***
    Copyright 2008-2012 Dr D Studios Pty Limited (ACN 127 184 954) (Dr. D Studios)

    This file is part of anim-studio-tools.

    anim-studio-tools is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    anim-studio-tools is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with anim-studio-tools.  If not, see <http://www.gnu.org/licenses/>.
***/
