/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: http://svn/drd/apps/bee/trunk/lib/gl/Program.cpp $"
 * SVN_META_ID = "$Id: Program.cpp 59286 2010-12-16 03:42:24Z stephane.bertout $"
 */

#include <GL/glxew.h>
#include <GL/glew.h>

#include "Program.h"
#include "Shader.h"
#include "../kernel/spam.h"
#include "../kernel/assert.h"
#include "../kernel/log.h"

using namespace bee;

//#define glpDEBUG
#ifdef glpDEBUG
	#define glpSPAM(t,s) SPAMt(t,s)
#else
	#define glpSPAM(t,s)
#endif

const char * Program::m_IncludeCode = NULL;

#undef LOG
#define LOG(CAT,X)

namespace
{
	const char *
	getTypeName( UInt a_Type )
	{
		switch ( a_Type )
		{
		case GL_FLOAT:
			return "Float";
		case GL_FLOAT_VEC2:
			return "FloatVec2";
		case GL_FLOAT_VEC3:
			return "FloatVec3";
		case GL_FLOAT_VEC4:
			return "FloatVec4";
		case GL_INT:
			return "Int";
		case GL_INT_VEC2:
			return "IntVec2";
		case GL_INT_VEC3:
			return "IntVec3";
		case GL_INT_VEC4:
			return "IntVec4";
		case GL_BOOL:
			return "Bool";
		case GL_BOOL_VEC2:
			return "BoolVec2";
		case GL_BOOL_VEC3:
			return "BoolVec3";
		case GL_BOOL_VEC4:
			return "BoolVec4";
		case GL_FLOAT_MAT2:
			return "FloatMat2";
		case GL_FLOAT_MAT3:
			return "FloatMat3";
		case GL_FLOAT_MAT4:
			return "FloatMat4";
		case GL_SAMPLER_1D:
			return "Sampler1D";
		case GL_SAMPLER_2D:
			return "Sampler2D";
		case GL_SAMPLER_3D:
			return "Sampler3D";
		case GL_SAMPLER_CUBE:
			return "SamplerCube";
		case GL_SAMPLER_1D_SHADOW:
			return "Sample1DShadow";
		case GL_SAMPLER_2D_SHADOW:
			return "Sampler2DShadow";
		}
		ASSERT( false && "All cases covered, what happened?" );
		return "ERROR!";
	}

}

void Program::init( 	const char* a_VertexShaderName,
						const char* a_FragmentShaderName,
						const char* a_GeometryShaderName,
						unsigned int a_GeomInType,
						unsigned int a_GeomOutType,
						bool a_UseIncludeFile )
{
	m_ID = glCreateProgram();

	bool do_geom = a_GeometryShaderName != NULL;

	Shader * vertShader = new Shader( GL_VERTEX_SHADER );
	Shader * fragShader = new Shader( GL_FRAGMENT_SHADER );
	Shader * geomShader = do_geom ? new Shader( GL_GEOMETRY_SHADER ) : NULL;

	bool vertShaderLoaded = vertShader->load( a_VertexShaderName );
	bool fragShaderLoaded = fragShader->load( a_FragmentShaderName );
	bool geomShaderLoaded = do_geom ? geomShader->load( a_GeometryShaderName ) : false;
	BOOST_ASSERT( vertShaderLoaded && fragShaderLoaded );
	if( do_geom ) BOOST_ASSERT( geomShaderLoaded );

	bool vertShaderCompiled = vertShader->compile( a_UseIncludeFile ? m_IncludeCode : NULL );
	bool fragShaderCompiled = fragShader->compile( a_UseIncludeFile ? m_IncludeCode : NULL );
	bool geomShaderCompiled = do_geom ? geomShader->compile( a_UseIncludeFile ? m_IncludeCode : NULL ) : false;
	BOOST_ASSERT( vertShaderCompiled );
	BOOST_ASSERT( fragShaderCompiled );
	if( do_geom ) BOOST_ASSERT( geomShaderCompiled );

	attachShader( vertShader );
	attachShader( fragShader );
	if( do_geom ) attachShader( geomShader );

	// geometry shader specific
	if( do_geom ){
		glProgramParameteriEXT( m_ID, GL_GEOMETRY_INPUT_TYPE_EXT, a_GeomInType );
		glProgramParameteriEXT( m_ID, GL_GEOMETRY_OUTPUT_TYPE_EXT, a_GeomOutType );

		int temp;
		glGetIntegerv( GL_MAX_GEOMETRY_OUTPUT_VERTICES_EXT, &temp );
		glProgramParameteriEXT( m_ID, GL_GEOMETRY_VERTICES_OUT_EXT, temp );
	}

	link();

	Int numUniforms = 0;
	Int maxChars = 0;
	glGetProgramiv( m_ID, GL_ACTIVE_UNIFORMS, (GLint *) &numUniforms );
	glGetProgramiv( m_ID, GL_ACTIVE_UNIFORM_MAX_LENGTH, (GLint *) &maxChars );
	if ( numUniforms > 0 )
	{
		LOG( DEBG, ( "Shader: %s/%s [%d]", a_VertexShaderName, a_FragmentShaderName, m_ID ) );
		char * name = new char[ maxChars + 1 ];
		for ( Int idx = 0 ; idx < numUniforms ; ++idx )
		{
			Int numChars = 0;
			UInt type = 0;
			Int size = 0;
			glGetActiveUniform( m_ID, idx, maxChars, (GLsizei*) &numChars, (GLsizei*) &size, (GLenum*) &type, name );

			Int location = glGetUniformLocation( m_ID, name );
			if ( location == -1 )
			{
				// built-in uniform
				continue;
			}

			LOG( DEBG, ( "\tuniform [%s]: [%s]", GetTypeName( type ), name ) );
		}
		delete[] name;
	}

	vertShader->deleteCode();
	fragShader->deleteCode();
	if( do_geom ) geomShader->deleteCode();
}

Program::Program( 	const char* a_VertexShaderName,
					const char* a_FragmentShaderName,
					bool a_UseIncludeFile )

{
	init( a_VertexShaderName, a_FragmentShaderName, NULL, 0, 0, a_UseIncludeFile );
}

Program::Program(	const char* a_VertexShaderName,
					const char* a_FragmentShaderName,
					const char* a_GeometryShaderName,
					unsigned int a_GeomInType,
					unsigned int a_GeomOutType,
					bool a_UseIncludeFile )
{
	init( a_VertexShaderName, a_FragmentShaderName, a_GeometryShaderName, a_GeomInType, a_GeomOutType, a_UseIncludeFile );
}

Program::~Program()
{
	glDeleteProgram( m_ID );
}

void Program::useIncludeFile( const char * a_IncludeShaderFile )
{
	m_IncludeCode = Shader::fileRead( a_IncludeShaderFile );
}

void Program::useIncludeString( const char * a_IncludeShaderString )
{
	m_IncludeCode = a_IncludeShaderString;
}

void Program::attachShader( Shader * shader )
{
	switch( shader->getType() )
	{
		case GL_VERTEX_SHADER:
			Assert( m_VertexShader.get() == NULL );
			m_VertexShader = SharedPtr< Shader >( shader );
			break;
		case GL_FRAGMENT_SHADER:
			Assert( m_FragmentShader.get() == NULL );
			m_FragmentShader = SharedPtr< Shader >( shader );
			break;
		case GL_GEOMETRY_SHADER:
			Assert( m_GeometryShader.get() == NULL );
			m_GeometryShader = SharedPtr< Shader >( shader );
			break;
		default:
			BOOST_ASSERT(0);
	}

	glAttachShader( m_ID, shader->getID() );
}

void Program::detachShader( Shader * shader )
{
	if ( shader->getType() == GL_VERTEX_SHADER )
	{
		Assert( m_VertexShader.get() == shader );
		m_VertexShader.reset();
	}
	else
	{
		Assert( m_FragmentShader.get() == shader );
		m_FragmentShader.reset();
	}

	glDetachShader( m_ID, shader->getID() );
}

bool Program::link()
{
	glLinkProgram( m_ID );

	int param;
	glGetProgramiv( m_ID, GL_LINK_STATUS, &param );
	if ( param ) return true;
	else return false;
}

void Program::use() const
{
	glUseProgram( m_ID );
}

void Program::release() const
{
	glUseProgram( 0 );
}

void Program::getInfoLog( String & o_InfoLog )
{
	int infologLength = 0;
	int charsWritten = 0;
	char *infoLog;

	glGetProgramiv( m_ID, GL_INFO_LOG_LENGTH, &infologLength );

	if ( infologLength > 0 )
	{
		infoLog = new char[ infologLength ];
		glGetShaderInfoLog( m_ID, infologLength, &charsWritten, infoLog );
		o_InfoLog = String( infoLog );
		delete[] infoLog;
	}
}

GLint Program::getUniformLocation( const char * name ) const
{
	return glGetUniformLocation( m_ID, name );
}

void Program::setUniform( 	const char * name,
							int val )
{
	GLint loc = glGetUniformLocation( m_ID, name );
	glUniform1i( loc, val );
}

void Program::setUniform( 	const char * name,
							const int* val,
							int varDim,
							int count )
{
	GLint loc = glGetUniformLocation( m_ID, name );
	if ( varDim == 4 ) glUniform4iv( loc, count, val );
	else if ( varDim == 3 ) glUniform3iv( loc, count, val );
	else if ( varDim == 2 ) glUniform2iv( loc, count, val );
	else if ( varDim == 1 ) glUniform1iv( loc, count, val );
}

void Program::setUniform( 	const char * name,
							float val )
{
	GLint loc = glGetUniformLocation( m_ID, name );
	glUniform1f( loc, val );
}

void Program::setUniformVec2( 	const char * name,
								const Vec2 & a_Value )
{
	GLint loc = glGetUniformLocation( m_ID, name );
	glUniform2fv( loc, 1, a_Value.getValue() );
}

void Program::setUniformVec3( 	const char * name,
								const Vec3 & a_Value )
{
	GLint loc = glGetUniformLocation( m_ID, name );
	glUniform3fv( loc, 1, a_Value.getValue() );
}

void Program::setUniformVec4( 	const char * name,
								const Vec4 & a_Value )
{
	GLint loc = glGetUniformLocation( m_ID, name );
	glUniform4fv( loc, 1, &a_Value.x );
}

void Program::setUniform( 	const char * name,
							const float* val,
							int varDim,
							int count )
{
	GLint loc = glGetUniformLocation( m_ID, name );
	if ( varDim == 4 ) glUniform4fv( loc, count, val );
	else if ( varDim == 3 ) glUniform3fv( loc, count, val );
	else if ( varDim == 2 ) glUniform2fv( loc, count, val );
	else if ( varDim == 1 ) glUniform1fv( loc, count, val );
}

void Program::setUniformMatrix( const char * name,
								const float * mat,
								bool bTranspose )
{
	GLint loc = glGetUniformLocation( m_ID, name );

	glUniformMatrix4fv( loc, 1, bTranspose, mat );
}

void Program::setUniformMatrix33( const char * name,
								const float * mat,
								bool bTranspose )
{
	GLint loc = glGetUniformLocation( m_ID, name );

	glUniformMatrix3fv( loc, 1, bTranspose, mat );
}

GLint Program::getAttribLocation( const char* name ) const
{
	return glGetAttribLocation( m_ID, name );
}

GLint Program::enableVertexAttribArray( const char * name )
{
	GLint loc = glGetAttribLocation( m_ID, name );
	glEnableVertexAttribArray( loc );
	return loc;
}

void Program::disableVertexAttribArray( GLint loc )
{
	glDisableVertexAttribArray( loc );
}

void Program::bindAttribLocation( 	unsigned int index,
									const char * name )
{
	glBindAttribLocation( m_ID, index, name );
	unsigned int err = glGetError();
	Assert( err == GL_NO_ERROR );
}

#undef glpDEBUG
#undef glpSPAM



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
