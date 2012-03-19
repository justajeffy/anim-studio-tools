/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: http://svn/drd/apps/bee/trunk/lib/io/textureLoader.cpp $"
 * SVN_META_ID = "$Id: textureLoader.cpp 107302 2011-10-11 01:24:53Z stephane.bertout $"
 */

#include "io/textureLoader.h"
#include "gl/Texture.h"
#include "gl/PBO.h"
#include "kernel/spam.h"
#include "kernel/assert.h"

#include <drdDebug/log.h>
#include <drdDebug/runtimeError.h>
#include <delightUtils/drdDelightUtils.h>

#include <iostream>
#include <boost/filesystem.hpp>

using namespace drd;
using namespace bee;
using namespace boost::filesystem;
using namespace std;

static bool s_FreeImageInited = false;

DRD_MKLOGGER( L, "drd.bee.io.textureLoader" );

//-------------------------------------------------------------------------------------------------
TextureLoader::TextureLoader( const string & i_FileName )
: m_Format( FIF_UNKNOWN )
, m_FIBitmap( NULL )
, m_Buffer( 0 )
, m_BufferSize( 0 )
, m_Width( 0 )
, m_Height( 0 )
{
	DRD_LOG_DEBUG( L, "trying to load file: " << i_FileName );

	std::string a_FileName;
	getExpandedCachePath( i_FileName, "texture", a_FileName );

	DRD_LOG_DEBUG( L, "loading resolved file: " << a_FileName );

	m_FileName = a_FileName;

	// First check that the file does exist !
	// note that getExpandedCachePath will return empty string if file not found
	if ( a_FileName.empty() )
		throw std::runtime_error( std::string( "TextureLoader: file doesn't exist " ) + i_FileName );

	const path filePath( a_FileName );
	string extension = filePath.extension();

	if ( extension == ".rat" ) // rat format not supported
	{
		// let's try a dds..
		string ddsFileName = a_FileName;
		string ratExt = ".rat";
		ddsFileName.replace( ddsFileName.find( ratExt ),
							 ratExt.size(),
							 ".dds" );

		if ( !exists( ddsFileName ) )
			throw std::runtime_error( std::string( "TextureLoader: ddsFileName doesn't exist " ) + ddsFileName );

		m_FileName = ddsFileName;
		m_Format = FIF_DDS;
	}
	else if ( extension == ".exr") // exr format not correctly supported by FreeImage
	{
		Imf::Rgba * pixelBuffer;

		Imf::RgbaInputFile in( a_FileName.c_str() );

		Imath::Box2i win = in.dataWindow();
		Imath::Box2i displayWin = in.displayWindow();

		Imath::V2i dim(win.max.x - win.min.x + 1, win.max.y - win.min.y + 1);
		Imath::V2i realdim(displayWin.max.x - displayWin.min.x + 1, displayWin.max.y - displayWin.min.y + 1);

		//SPAM( dim.x ); SPAM( dim.y );
		pixelBuffer = new Imf::Rgba[realdim.x * realdim.y];
		m_BufferSize = realdim.x * realdim.y * sizeof( Imf::Rgba );
		memset(pixelBuffer, 0, m_BufferSize);

		int dx = displayWin.min.x;
		int dy = displayWin.min.y;

		in.setFrameBuffer(pixelBuffer - dx - dy * dim.x, 1, dim.x);
		in.readPixels(win.min.y, win.max.y);

		// invert Y on the texture as for some reason it comes inverted...
		Imf::Rgba * invPixelBuffer = new Imf::Rgba[realdim.x * realdim.y];
		for (int i = 0; i < realdim.y; ++i)
		{
			memcpy( &invPixelBuffer[ i * realdim.x ], &pixelBuffer[ ((realdim.y-1) - i) * realdim.x ], sizeof(Imf::Rgba) * realdim.x );
		}
		delete [] pixelBuffer;

		m_Format = FIF_EXR;

		m_Width = realdim.x;
		m_Height = realdim.y;
		m_Buffer = invPixelBuffer;
		m_TextureFormat = Texture::eRGBA16F;
	}
	else if ( extension == ".dds")
	{
		m_Format = FIF_DDS;
	}
	else
	{
		// init Free Image only once
		if ( !s_FreeImageInited )
		{
			FreeImage_Initialise();
			s_FreeImageInited = true;
		}

		const char * texFileName = a_FileName.c_str();

		m_Format = FreeImage_GetFileType( texFileName );
		m_FIBitmap = FreeImage_Load( m_Format, texFileName );

		FIBITMAP * temp = m_FIBitmap;
		m_FIBitmap = FreeImage_ConvertTo32Bits( m_FIBitmap );
		FreeImage_Unload( temp );

		m_Width = FreeImage_GetWidth( m_FIBitmap );
		m_Height = FreeImage_GetHeight( m_FIBitmap );
		m_Buffer = FreeImage_GetBits( m_FIBitmap );
		m_BufferSize = m_Width * m_Height * 4 * sizeof( unsigned char );
		m_TextureFormat = Texture::eBGRA;
	}
}

//-------------------------------------------------------------------------------------------------
TextureLoader::~TextureLoader()
{
	unload();
}

//-------------------------------------------------------------------------------------------------
void TextureLoader::unload()
{
	if ( m_FIBitmap != NULL ) FreeImage_Unload( m_FIBitmap );
	m_FIBitmap = NULL;

	if ( m_Format == FIF_EXR )
	{
		delete[] ( (Imf::Rgba *) m_Buffer );
	}
}

//-------------------------------------------------------------------------------------------------
SharedPtr< Texture > TextureLoader::createTexture()
{
	if ( m_Format == FIF_UNKNOWN ) return SharedPtr< Texture >();

	if ( m_Format == FIF_DDS )
	{
		return SharedPtr<Texture>( DDSLoader::read( m_FileName.c_str() ) );
	}

	SharedPtr<Texture> pTexture = SharedPtr<Texture>( new Texture( m_Width, m_Height, m_TextureFormat, Texture::e2D ) );
	pTexture->init( m_Buffer );

	return pTexture;
}

//-------------------------------------------------------------------------------------------------
SharedPtr< PBO > TextureLoader::createPBO()
{
	if( m_Format == FIF_UNKNOWN )
	{
		SharedPtr< PBO > temp = SharedPtr< PBO >();
		return temp;
	}
	if( m_Format == FIF_DDS )
	{
		ASSERT( false && "DDS not currently supported for PBOs" );
		SharedPtr< PBO > temp = SharedPtr< PBO >();
		return temp;
	}


	SharedPtr< PBO > pPBO = SharedPtr< PBO >( new PBO( m_Width, m_Height, m_Buffer, m_BufferSize ) );

	return pPBO;
}

//-------------------------------------------------------------------------------------------------
const std::string s_FormatNames[] =
{
		"BMP",
		"ICO",
		"JPEG",
		"JNG",
		"KOALA",
		"LBM/IFF",
		"MNG",
		"PBM",
		"PBMRAW",
		"PCD",
		"PCX",
		"PGM",
		"PGMRAW",
		"PNG",
		"PPM",
		"PPMRAW",
		"RAS",
		"TARGA",
		"TIFF",
		"WBMP",
		"PSD",
		"CUT",
		"XBM",
		"XPM",
		"DDS",
		"GIF",
		"HDR",
		"FAXG3",
		"SGI",
		"EXR",
		"J2K",
		"JP2",
};

//-------------------------------------------------------------------------------------------------
void TextureLoader::reportStats()
{
	DRD_LOG_DEBUG( L, "Texture loaded : " << m_FileName );

	/*cout << "TextureLoader Report :" << endl;
	cout << "- FileName : " << m_FileName.c_str() << endl;
	cout << "- Width : " << m_Width << endl;
	cout << "- Height : " << m_Height << endl;
	if ( m_Format == FIF_UNKNOWN ) cout << "- Format : UNKNOWN";
	else cout << "- Format : " << s_FormatNames[ m_Format ] << endl;*/
}

//-------------------------------------------------------------------------------------------------

// DDS Loader

namespace
{
	const unsigned int DDS_PFMASK		= 0x0000004F;
	const unsigned int DDS_LUMINANCE8 = 0x00000000;
	const unsigned int DDS_FOURCC		= 0x00000004;
	const unsigned int DDS_RGB		= 0x00000040;
	const unsigned int DDS_RGBA		= 0x00000041;
	const unsigned int DDS_DEPTH		= 0x00800000;

	const unsigned int DDS_COMPLEX = 0x00000008;
	const unsigned int DDS_CUBEMAP = 0x00000200;
	const unsigned int DDS_VOLUME  = 0x00200000;

	const unsigned int FOURCC_DXT1 = 0x31545844; //(MAKEFOURCC('D','X','T','1'))
	const unsigned int FOURCC_DXT3 = 0x33545844; //(MAKEFOURCC('D','X','T','3'))
	const unsigned int FOURCC_DXT5 = 0x35545844; //(MAKEFOURCC('D','X','T','5'))
}

unsigned int clampSize(unsigned int size)
{
	if (size == 0)
	{
		return (1);
	}

	return (size);
}

Texture * TextureLoader::DDSLoader::read( const string & a_FileName )
{
	FILE * file = fopen( a_FileName.c_str(), "r" );

	char ExtCode[ 4 ];
	fread( ExtCode, sizeof(ExtCode), 1, file );

	if ( strncmp( ExtCode, "DDS ", 4 ) != 0 )
	{
		return NULL;
	}

	DDSHeader sDdsHeader;
	fread( &sDdsHeader, sizeof(sDdsHeader), 1, file );

	bool bCubeMap = false, bVolume = false, bCompressed;
    if ( sDdsHeader.dwCaps2 & DDS_CUBEMAP )										bCubeMap = true;
	if ( ( sDdsHeader.dwCaps2 & DDS_VOLUME ) && ( sDdsHeader.dwDepth > 0 ) )	bVolume = true;

	Texture::Format eFormat;
    if ( sDdsHeader.ddspf.dwFlags & DDS_FOURCC )
    {
        switch( sDdsHeader.ddspf.dwFourCC )
        {
            case FOURCC_DXT1:
				eFormat = Texture::eDXTC1;
                bCompressed = true;
                break;

            case FOURCC_DXT3:
                eFormat = Texture::eDXTC3;
                bCompressed = true;
                break;

            case FOURCC_DXT5:
                eFormat = Texture::eDXTC5;
                bCompressed = true;
                break;

			case 36:
				eFormat = Texture::eRGBA16F;
				bCompressed = false;
				break;

			case 116:
				eFormat = Texture::eRGBA32F;
				bCompressed = false;
				break;

            default:
                return NULL;
        }
    }
    else if ( sDdsHeader.ddspf.dwFlags == DDS_RGBA && sDdsHeader.ddspf.dwRGBBitCount == 32 )
    {
		eFormat = Texture::eRGBA;
        bCompressed = false;
	}
    /*else if (sDdsHeader.ddspf.dwFlags == DDS_RGB  && sDdsHeader.ddspf.dwRGBBitCount == 32)
    {
        eFormat = Texture::X8R8G8B8;
        bCompressed = false;
    }
    else if (sDdsHeader.ddspf.dwFlags == DDS_RGB  && sDdsHeader.ddspf.dwRGBBitCount == 24)
    {
		zError("TODO");
        eFormat = Texture::X8R8G8B8;
        bCompressed = false;
    }
	else if ((sDdsHeader.ddspf.dwFlags & DDS_PFMASK) == DDS_LUMINANCE8  && sDdsHeader.ddspf.dwRGBBitCount == 8)
    {
        eFormat = Texture::L8;
        bCompressed = false;
    }*/
	else
	{
        return NULL;
    }

    unsigned int nWidth  = sDdsHeader.dwWidth;
    unsigned int nHeight = sDdsHeader.dwHeight;
    unsigned int nDepth  = sDdsHeader.dwDepth;
	unsigned int nMipMapCount = sDdsHeader.dwMipMapCount;

	Texture * pTexture = NULL;

	Assert( ( nWidth == nHeight ) && "todo texture rect!" );

	if ( bVolume )
	{
		Assert( bCompressed == false );
		pTexture = new Texture( nWidth, nHeight, nDepth, eFormat, Texture::e3D, nMipMapCount );
	}
	else if ( bCubeMap )
	{
		Assert( nWidth == nHeight );
		pTexture = new Texture( nWidth, eFormat, Texture::eCUBE, nMipMapCount );
		nDepth = 6;
	}
	else
	{
		pTexture = new Texture( nWidth, nHeight, eFormat, Texture::e2D, nMipMapCount );
		nDepth = 1;
	}

	pTexture->init();

	char * o_Buffer;
	char * o_AlignedBuffer;
	int o_LockedSize;

	// load all surfaces for the image (6 surfaces for cubemaps)
	for ( unsigned int nCubeFace = 0 ; nCubeFace < nDepth ; nCubeFace++ )
	{
		pTexture->lock( 0, nCubeFace, o_Buffer, o_AlignedBuffer, o_LockedSize );
		fread( o_AlignedBuffer, sizeof(char), o_LockedSize, file );
		pTexture->unlock( 0, nCubeFace, o_Buffer, o_AlignedBuffer, o_LockedSize );

		int w = clampSize( nWidth >> 1 );
		int h = clampSize( nHeight >> 1 );

		// load all mipmaps for current surface
		for ( unsigned int nMipMapLevel = 1 ; nMipMapLevel < nMipMapCount && ( w || h ) ; ++nMipMapLevel )
		{
			pTexture->lock( nMipMapLevel, nCubeFace, o_Buffer, o_AlignedBuffer, o_LockedSize );
			fread( o_AlignedBuffer, sizeof(char), o_LockedSize, file );
			pTexture->unlock( nMipMapLevel, nCubeFace, o_Buffer, o_AlignedBuffer, o_LockedSize );

			// shrink to next power of 2
			w = clampSize( w >> 1 );
			h = clampSize( h >> 1 );
		}
	}

	return pTexture;
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
