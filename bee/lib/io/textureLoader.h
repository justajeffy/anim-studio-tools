/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: http://svn/drd/apps/bee/trunk/lib/io/textureLoader.h $"
 * SVN_META_ID = "$Id: textureLoader.h 27186 2010-04-07 01:35:04Z david.morris $"
 */

#ifndef bee_textureLoader_h
#define bee_textureLoader_h
#pragma once

#include "../kernel/types.h"
#include "../kernel/smartPointers.h"
#include "../gl/Texture.h"

#include <boost/algorithm/string.hpp>
#include <FreeImage.h>
#include <OpenEXR/ImfRgbaFile.h>

namespace bee
{
	class Texture;
	class PBO;

	//-------------------------------------------------------------------------------------------------
	//! TextureLoader is a IO utility class used to load different texture format (jpeg, dds, exr, tiff, png, tga..)
	class TextureLoader
	{
		//! DDSLoader is a IO::TextureLoader utility sub-class used to load DDS texture format
		class DDSLoader
		{
			friend class TextureLoader;

			//! DDS Pixel Format
			struct DDSPixelFormat
			{
				unsigned int	dwSize;
				unsigned int	dwFlags;
				unsigned int	dwFourCC;
				unsigned int	dwRGBBitCount;
				unsigned int	dwRBitMask;
				unsigned int	dwGBitMask;
				unsigned int	dwBBitMask;
				unsigned int	dwABitMask;
			};

			//! DDS Header
			struct DDSHeader
			{
				unsigned int	dwSize;
				unsigned int	dwFlags;
				unsigned int	dwHeight;
				unsigned int	dwWidth;
				unsigned int	dwPitchOrLinearSize;
				unsigned int	dwDepth;
				unsigned int	dwMipMapCount;
				unsigned int	dwReserved1[11];
				DDSPixelFormat 	ddspf;
				unsigned int	dwCaps1;
				unsigned int	dwCaps2;
				unsigned int	dwReserved2[3];
			};

			//! Function to create the Texture from a specified DDS
			static bee::Texture * read( const std::string & a_FileName );
		};

	public:
		//! Constructor (OpenXR used for exr format, DDSLoader for DDS and FreeImage for all the other format)
		TextureLoader( const std::string & a_FileName );

		//! Destructor
		~TextureLoader();

		//! Unload and free the memory used
		void unload();
		//! Returns the created texture
		SharedPtr<Texture> createTexture();
		SharedPtr<PBO> createPBO();

		//! Reports some stats
		void reportStats();

		void* getBuffer( void ) const
		{
			return m_Buffer;
		}

		UInt getWidth() const
		{
			return m_Width;
		}
		UInt getHeight() const
		{
			return m_Height;
		}

	private:
		std::string m_FileName;

		FREE_IMAGE_FORMAT m_Format;
		FIBITMAP * m_FIBitmap;
		void * m_Buffer;
		UInt m_BufferSize;
		Texture::Format m_TextureFormat;

		UInt m_Width,
		m_Height;
	};
}

#endif // bee_textureLoader_h


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
