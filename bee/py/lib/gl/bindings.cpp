/*
 * bindings.cpp
 *
 *  Created on: Oct 27, 2009
 *      Author: stephane.bertout
 */


#include <boost/python.hpp>
using namespace boost::python;

#include <kernel/pythonHelper.h>
#include <kernel/assert.h>
#include <gl/Renderer.h>
#include <gl/Shader.h>
#include <gl/Program.h>
#include <gl/Texture.h>
#include <gl/glError.h>
#include "PyProgram.h"
#include "PyTexture.h"
#include "PyRenderTarget.h"
#include <io/TextureTools.h>

using namespace bee;

void useLutTextureOnProgram(const Renderer & a_Renderer, const py::Program & a_Program, int idx )
{
	a_Renderer.getLutTexture()->use( idx, a_Program.getProgram() );
}

void importGlBindings()
{
	def( "SaveGLScreenShot", SaveGLScreenShot );
	def( "initGL", Renderer::init );
	def( "ProgramUseIncludeString", Program::useIncludeString );
	def( "useLutTextureOnProgram", useLutTextureOnProgram );

    enum_<Shader::Type>("ShaderType")
		.value( "Vertex",   Shader::eVertex )
		.value( "Fragment", Shader::eFragment )
		.value( "Geometry", Shader::eGeometry )
		;

	class_< py::Program >( "Program" )
		.def( "read", (void ( py::Program::* )( const std::string&, const std::string& ) )( &py::Program::read ), ( arg("i_VertexShaderPath"), arg("i_FragmentShaderPath") ) )
		.def( "read", (void ( py::Program::* )( const std::string&, const std::string&, const std::string&, unsigned int, unsigned int ) )( &py::Program::read )
					, ( arg("i_VertexShaderPath"), arg("i_FragmentShaderPath"), arg("i_GeometryShaderPath"), arg("i_GeomInType"), arg("i_GeomOutType") ) )
		.def( "use", &py::Program::use )
		.def( "release", &py::Program::release )
		.def( "setUniform", (void ( py::Program::* )(const std::string&,float)) (&py::Program::setUniform)
						  , (arg("x")) )
		.def( "setUniform", (void ( py::Program::* )(const std::string&,float,float)) (&py::Program::setUniform)
						  , (arg("x"), arg("y")) )
		.def( "setUniform", (void ( py::Program::* )(const std::string&,float,float,float)) (&py::Program::setUniform)
						  , (arg("x"), arg("y"), arg("z")) )
		.def( "setUniform", (void ( py::Program::* )(const std::string&,float,float,float,float)) (&py::Program::setUniform)
						  , (arg("x"), arg("y"), arg("z"), arg("w")) )
	;

	class_< py::Texture >( "Texture" )
		.def( "read", &py::Texture::read )
		.def( "use", &py::Texture::use )
		.def( "release", &py::Texture::release )
	;

	class_< Renderer >( "Renderer" )
		.def( "initLut", &Renderer::initLut )
	;

	class_< py::RenderTarget >( "RenderTarget" )
		.def( "create", &py::RenderTarget::create )
		.def( "destroy", &py::RenderTarget::destroy )
		.def( "use", &py::RenderTarget::use )
		.def( "release", &py::RenderTarget::release )
		.def( "getTexture", &py::RenderTarget::getTexture, return_value_policy<manage_new_object>() )
		.def( "getWidth", &py::RenderTarget::getWidth )
		.def( "getHeight", &py::RenderTarget::getHeight )
	;
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
