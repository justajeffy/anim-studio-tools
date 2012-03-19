/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: http://svn/drd/apps/bee/trunk/lib/gl/Texture.h $"
 * SVN_META_ID = "$Id: Texture.h 58616 2010-12-13 04:32:38Z stephane.bertout $"
 */

#ifndef bee_Texture_h
#define bee_Texture_h
#pragma once

#include <GL/glew.h>
#include <GL/glut.h>
#include "../kernel/types.h"

namespace bee
{
	class Program;

	//-------------------------------------------------------------------------------------------------
	//! Texture is a GL utility class used to manipulate texture (1d, 2d, 3d, cube)
	class Texture
	{
	public:
		//! Texture Format
		enum Format
		{
			//! Blue Green Red Alpha
			eBGRA = 0,
			//! Blue Green Red
			eBGR,
			//! Red Green Blue Alpha
			eRGBA,
			//! Red Green Blue
			eRGB,
			//! Half Float Format
			eRGBA16F,
			//! Half Float Format
			eRGB16F,
			//! Float Format
			eRGB32F,
			//! Float Format
			eRGBA32F,

			//! 16b Depth Format
			eDepth16,
			//! 24b Depth Format
			eDepth24,

			//! S3TC / DXT1 Compression Format
			eDXTC1,
			//! S3TC / DXT3 Compression Format
			eDXTC3,
			//! S3TC / DXT5 Compression Format
			eDXTC5,

			eL16F,
			eL32F,

			eNone,
		};

		//! Texture Type
		enum Type
		{
			e1D = GL_TEXTURE_1D,
			e2D = GL_TEXTURE_2D,
			e2Dr = 0x84F5, // GL_TEXTURE_RECTANGLE_ARB,
			e3D = GL_TEXTURE_3D,
			eCUBE = GL_TEXTURE_CUBE_MAP,
		};
		//! Maximum Texture Count Supported inside a shader
		enum Constants
		{
			MaxTextureCount = 24,
		};

		//! 1D Constructor
		Texture( 	UInt m_Size,
					Format a_Format,
					Type a_Type,
					UInt a_MipMapCount = 0 );
		//! 2D Constructor
		Texture( 	UInt m_Width,
					UInt m_Height,
					Format a_Format,
					Type a_Type,
					UInt a_MipMapCount = 0 );
		//! 3D Constructor
		Texture( 	UInt m_Width,
					UInt m_Height,
					UInt m_Depth,
					Format a_Format,
					Type a_Type,
					UInt a_MipMapCount = 0 );
		//! Destructor
		~Texture();

		//! Init function
		void init( const void * Buffer = NULL, bool a_ForceTexImage2D = false );
		//! Setup GL State
		void use( 	UInt a_Idx ) const;
		void use( 	UInt a_Idx,
					const Program * a_Program,
					bool a_SetUniformTexSize = false,
					int a_Location = -1 ) const;
		//! Unset the GL State
		void release( UInt a_Idx ) const;

		//! Lock the specified mipmap/cubeface
		char * lock( 	int nMipMapLevels,
						int nCubeFace,
						char* & o_Buffer,
						char* & o_AlignedBuffer,
						int & o_LockedSize );
		//! Unlock the specified mipmap/cubeface
		void unlock( 	int nMipMapLevels,
						int nCubeFace,
						char* & o_Buffer,
						char* & o_AlignedBuffer,
						int & o_LockedSize );

		//! Returns the Width of the texture
		UInt getWidth() const
		{
			return m_Width;
		}
		//! Returns the Height of the texture
		UInt getHeight() const
		{
			return m_Height;
		}
		//! Returns the Depth of the texture (only valid for 3D Texture)
		UInt getDepth() const
		{
			return m_Depth;
		}

		//! get the gl texture id
		UInt getId() const;

		static const char * getTextureSamplersName( UInt a_Unit );
		static const char * getTextureSamplerSizesName( UInt a_Unit );

	private:
		friend class RenderTarget;

		void initOglInfos();
		bool isFormatCompressed() const
		{
			return ( m_Format == eDXTC1 ) || ( m_Format == eDXTC3 ) || ( m_Format == eDXTC5 );
		}
		int computeSize( int nMipMapLevels );

		UInt m_GLId;

		UInt m_Width;
		UInt m_Height;
		UInt m_Depth;

		Format m_Format;
		Type m_Type;
		UInt m_MipMapCount;

		UInt m_OglFormat;
		UInt m_OglInternalFormat;
		UInt m_OglPixelDataType;
	};
}
#endif // bee_Texture_h


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
