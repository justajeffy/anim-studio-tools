/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: $"
 * SVN_META_ID = "$Id: program.cpp 42544 2010-08-17 04:31:03Z allan.johns $"
 */

//-------------------------------------------------------------------------------------------------
#include "program.h"
#include "log.h"

// renderman
#include "rx.h"

// bee includes
#include <bee/gl/Program.h>

#include <stdexcept>

//-------------------------------------------------------------------------------------------------
using namespace grind;

//-------------------------------------------------------------------------------------------------
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
		_program.reset( new bee::Program( i_VertexShaderPath.c_str(), i_FragmentShaderPath.c_str() ) );
		if( _program ) return;
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
		_program.reset( new bee::Program( i_VertexShaderPath.c_str(),
		                                  i_FragmentShaderPath.c_str(),
		                                  i_GeometryShaderPath.c_str(),
		                                  i_GeomInType,
		                                  i_GeomOutType ) );
		if( _program ) return;
	} catch ( ... ) {}

	throw std::runtime_error( std::string("error loading GLSL shader program from ")+i_VertexShaderPath+" and "+i_FragmentShaderPath+" and "+i_GeometryShaderPath );
}

//-------------------------------------------------------------------------------------------------
void Program::use()
{
	if( !_program )
		throw std::runtime_error( "Program::use() failed as program hasn't been set up correctly" );

	_program->use();
}

//-------------------------------------------------------------------------------------------------
void Program::unUse()
{
	_program->release();
}

//-------------------------------------------------------------------------------------------------
bee::Program* Program::getProgram() const
{
	if( !_program )
		throw std::runtime_error( "Program::getProgram() failed as program hasn't been set up correctly" );

	return _program.get();
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
