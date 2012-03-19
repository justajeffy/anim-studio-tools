/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: http://svn/drd/apps/bee/trunk/py/lib/gl/PyTexture.h $"
 * SVN_META_ID = "$Id: PyTexture.h 17883 2009-11-30 03:36:10Z david.morris $"
 */

#ifndef bee_py_PyTexture_h
#define bee_py_PyTexture_h

#include <string>
#include <boost/shared_ptr.hpp>
#include <kernel/types.h>

//----------------------------------------------------------------------------
namespace bee
{
	class Texture;
	//----------------------------------------------------------------------------
	namespace py
	{
		//----------------------------------------------------------------------------
		class Program;
		//----------------------------------------------------------------------------
		//! a texture map
		class Texture
		{
		public:
			//! default constructor
			Texture();

			//! default constructor
			Texture( const bee::Texture * a_rawTexture );

			//! load from a file path
			Texture( const std::string& a_Path );

			//! default destructor
			~Texture();

			//! load from a file path
			void read( const std::string& a_Path );

			//! rotate the texture using cuda
			void rotate();

			//! dump using gl2 calls
			void dumpGL( float lod );

			//! use with a gl shader
			void use( 	UInt idx,
						const Program& program,
						bool setUniformTexSize = false,
						int location = -1 ) const;

			//! stop using the specified texture
			void release( UInt idx ) const;

		private:

			//! the underlying OpenGL texture, this one is refcounted for python's sake
			boost::shared_ptr< bee::Texture > m_refCountTex;

			//! if this texture is null, use m_refCountTex, else use this one. this one
			//! is owned by someone else, so it doesn't get refcounted
			const bee::Texture * m_rawTex;
		};

	} // namespace py
} // namespace bee1

#endif // bee_py_PyTexture_h


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
