/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: http://svn/drd/apps/bee/trunk/py/lib/gl/PyProgram.cpp $"
 * SVN_META_ID = "$Id: PyProgram.cpp 30989 2010-05-11 23:23:13Z chris.cooper $"
 */

//----------------------------------------------------------------------------
// system includes
#include <stdexcept>
// bee includes
#include <gl/Program.h>
#include <GL/glew.h>
#include <GL/glut.h>
// bee::py includes
#include "PyProgram.h"

//----------------------------------------------------------------------------
using namespace bee::py;

//----------------------------------------------------------------------------
Program::Program()
{
}

//-------------------------------------------------------------------------------------------------
Program::~Program()
{
}

//-------------------------------------------------------------------------------------------------
void Program::read( const std::string& i_VertexShaderPath, const std::string& i_FragmentShaderPath )
{
	try {
		m_program.reset( new bee::Program( i_VertexShaderPath.c_str(), i_FragmentShaderPath.c_str() ) );
		if( m_program ) return;
	} catch ( ... ) {}

	throw std::runtime_error( std::string("error loading GLSL shader program from ")+i_VertexShaderPath+" and "+i_FragmentShaderPath );
}

//-------------------------------------------------------------------------------------------------
void Program::read(	const std::string& i_VertexShaderPath,
					const std::string& i_FragmentShaderPath,
					const std::string& i_GeometryShaderPath,
					unsigned int i_GeomInType,
					unsigned int i_GeomOutType )
{
	try {
		m_program.reset( new bee::Program( i_VertexShaderPath.c_str(),
		                                   i_FragmentShaderPath.c_str(),
		                                   i_GeometryShaderPath.c_str(),
		                                   i_GeomInType,
		                                   i_GeomOutType ) );
		if( m_program ) return;
	} catch ( ... ) {}

	throw std::runtime_error( std::string("error loading GLSL shader program from ")+i_VertexShaderPath+" and "+i_FragmentShaderPath+" and "+i_GeometryShaderPath );
}

//----------------------------------------------------------------------------
void Program::use()
{
	if( !m_program )
		throw std::runtime_error( "Program::use() failed as program hasn't been set up correctly" );

	m_program->use();
}

//----------------------------------------------------------------------------
void Program::release()
{
	m_program->release();
}

//----------------------------------------------------------------------------
bee::Program* Program::getProgram() const
{
	return m_program.get();
}

//-------------------------------------------------------------------------------------------------
void Program::setUniform( const std::string& name,
					 	  float x )
{
	bee::Program * program = getProgram();
	program->setUniform( name.c_str(), x );
}

//-------------------------------------------------------------------------------------------------
void Program::setUniform( const std::string& name,
					 	  float x, float y )
{
	bee::Program * program = getProgram();
	program->setUniformVec2( name.c_str(), Vec2(x,y) );
}

//-------------------------------------------------------------------------------------------------
void Program::setUniform( const std::string& name,
					 	  float x, float y, float z )
{
	bee::Program * program = getProgram();
	program->setUniformVec3( name.c_str(), Vec3(x,y,z) );
}

//-------------------------------------------------------------------------------------------------
void Program::setUniform( const std::string& name,
					 	  float x, float y, float z, float w )
{
	bee::Program * program = getProgram();
	program->setUniformVec4( name.c_str(), Vec4(x,y,z, w) );
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
