/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: $"
 * SVN_META_ID = "$Id: texture.h 42544 2010-08-17 04:31:03Z allan.johns $"
 */

#ifndef grind_texture_h
#define grind_texture_h

//-------------------------------------------------------------------------------------------------

#include <string>
#include <boost/shared_ptr.hpp>

//-------------------------------------------------------------------------------------------------
namespace bee
{
	class Texture;
}

namespace grind
{

class Program;


//-------------------------------------------------------------------------------------------------
//! a texture map
class Texture
{
	//! current state of the texture object
	enum State { OPEN_GL, CUDA };

public:
	//! default constructor
	Texture();

	//! load from a file path
	void read( const std::string& path );

	//! use with a GLSL shader program
	void use( unsigned int idx, const Program& program, bool setUniformTexSize = false, int location = -1 ) const;

	//! stop using the specified texture
	void unUse( unsigned int idx ) const;

	//! @cond DEV

	//! load from a file path
	Texture( const std::string& path );

	//! default destructor
	~Texture();

private:

	//! the underlying OpenGL texture
	boost::shared_ptr< bee::Texture > m_Tex;

	//! pixel buffer object required for cuda interop
	unsigned int m_Pbo;

	//! current state of texture
	mutable State m_State;

	//! pointer to device data for cuda processing
	mutable unsigned int* m_DevicePtr;

	//! has a texture been allocated?
	bool m_Allocated;

	//! initialization shared between constructors
	void init();

	//! indicate that we're about to do some cuda processing (optional)
	void prepForCuda() const;

	//! indicate that we're about to do some GL drawing (optional)
	void prepForGL() const;

	//! rotate the texture using cuda
	void rotate();

	//! dump using gl2 calls
	void dumpGL( float lod );

	unsigned int* getDevicePtr() const;



	//! @endcond
};

} // namespace Grind

#endif /* grind_texture_h */


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
