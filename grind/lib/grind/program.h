/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: $"
 * SVN_META_ID = "$Id: program.h 42544 2010-08-17 04:31:03Z allan.johns $"
 */

#ifndef grind_program_h
#define grind_program_h

//-------------------------------------------------------------------------------------------------
#include <string>
#include <boost/shared_ptr.hpp>

//-------------------------------------------------------------------------------------------------
// pre-declare
namespace bee
{
	class Program;
}


namespace grind
{

//-------------------------------------------------------------------------------------------------
//! An OpenGL GLSL shader program
class Program
{
public:
	//! default constructor
	Program();

	//! load from a file path
	void read(	const std::string& i_VertexShaderPath,
				const std::string& i_FragmentShaderPath );

	//! load from a file path
	void read(	const std::string& i_VertexShaderPath,
				const std::string& i_FragmentShaderPath,
				const std::string& i_GeometryShaderPath,
				unsigned int i_GeomInType,
				unsigned int i_GeomOutType );

	//! use within an OpenGL drawing context
	void use();

	//! un-use (when finished drawing)
	void unUse();

	//! @cond DEV

	//! destructor
	~Program();

	//! to pass to a texture
	bee::Program* getProgram() const;

private:

	//! the underlying bee texture
	boost::shared_ptr< bee::Program > _program;

	//! @endcond
};

} // namespace Grind

#endif /* grind_program_h */


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
