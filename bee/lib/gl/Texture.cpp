/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: http://svn/drd/apps/bee/trunk/lib/gl/Texture.cpp $"
 * SVN_META_ID = "$Id: Texture.cpp 58616 2010-12-13 04:32:38Z stephane.bertout $"
 */

#include <GL/glxew.h>
#include <GL/glew.h>

#include "Texture.h"
#include "Program.h"
#include "glError.h"
#include "glext.h"
#include "glExtensions.h"

#include "../kernel/spam.h"
#include "../kernel/assert.h"
#include "../kernel/log.h"

#include <stdio.h>

using namespace bee;

namespace
{
	//-------------------------------------------------------------------------------------------------
	char s_BytePixelSizeOGL[ Texture::eNone ] =
	{
			4, 	//eBGRA = 0,
			4, 	//eBGR,
			4, 	//eRGBA,
			4, 	//eRGB,
			8, 	//eRGBA16F,
			8, 	//eRGB16F,
			16, //eRGBA32F,
			16, //eRGB32F,
			2, //eDepth16,
			3, //eDepth24,
			1, //eDXTC1,
			2, //eDXTC3,
			2, //eDXTC5,
			2, //eL16F
			4, //eL32F
	};

	//-------------------------------------------------------------------------------------------------
	const char * sTextureSamplers[ Texture::MaxTextureCount  ] = {
										"TextureSampler0",
										"TextureSampler1",
										"TextureSampler2",
										"TextureSampler3",
										"TextureSampler4",
										"TextureSampler5",
										"TextureSampler6",
										"TextureSampler7",
										"TextureSampler8",
										"TextureSampler9",
										"TextureSampler10",
										"TextureSampler11",
										"TextureSampler12",
										"TextureSampler13",
										"TextureSampler14",
										"TextureSampler15",
										"TextureSampler16",
										"TextureSampler17",
										"TextureSampler18",
										"TextureSampler19",
										"TextureSampler20",
										"TextureSampler21",
										"TextureSampler22",
										"TextureSampler23",
										};
	//-------------------------------------------------------------------------------------------------
	const char * sTextureSamplerSizes[ Texture::MaxTextureCount  ] = {
										"TextureSamplerSize0",
										"TextureSamplerSize1",
										"TextureSamplerSize2",
										"TextureSamplerSize3",
										"TextureSamplerSize4",
										"TextureSamplerSize5",
										"TextureSamplerSize6",
										"TextureSamplerSize7",
										"TextureSamplerSize8",
										"TextureSamplerSize9",
										"TextureSamplerSize10",
										"TextureSamplerSize11",
										"TextureSamplerSize12",
										"TextureSamplerSize13",
										"TextureSamplerSize14",
										"TextureSamplerSize15",
										"TextureSamplerSize16",
										"TextureSamplerSize17",
										"TextureSamplerSize18",
										"TextureSamplerSize19",
										"TextureSamplerSize20",
										"TextureSamplerSize21",
										"TextureSamplerSize22",
										"TextureSamplerSize23",
										};


	//-------------------------------------------------------------------------------------------------
	unsigned int	sizeCompressed(unsigned int nWidth, unsigned int nHeight, Texture::Format eFormat)
	{
		return ( ( ( nWidth + 3 ) / 4 ) * ( ( nHeight + 3 ) / 4 ) * ( ( eFormat == Texture::eDXTC1 ) ? ( 8 ) : ( 16 ) ) );
	}

	//-------------------------------------------------------------------------------------------------
	unsigned int	size(unsigned int nWidth, unsigned int nHeight, Texture::Format eFormat)
	{
		return nWidth * nHeight * s_BytePixelSizeOGL[ eFormat ];
	}
}

//-------------------------------------------------------------------------------------------------
Texture::Texture( 	UInt a_Size,
					Format a_Format,
					Type a_Type,
					UInt a_MipMapCount )
: m_GLId( 0 )
, m_Width( a_Size )
, m_Height( 1 )
, m_Depth( 0 )
, m_Format( a_Format )
, m_Type( a_Type )
, m_MipMapCount( a_MipMapCount )
{
	initOglInfos();

	Assert( ( m_Type == e1D || m_Type == eCUBE ) && "Wrong texture type" );

	if ( m_Type == eCUBE ) m_Height = m_Width;
}

//-------------------------------------------------------------------------------------------------
Texture::Texture( 	UInt a_Width,
					UInt a_Height,
					Format a_Format,
					Type a_Type,
					UInt a_MipMapCount )
: m_GLId( 0 )
, m_Width( a_Width )
, m_Height( a_Height )
, m_Depth( 0 )
, m_Format( a_Format )
, m_Type( a_Type )
, m_MipMapCount( a_MipMapCount )
{
	initOglInfos();

	Assert( ( m_Type == e2D || m_Type == e2Dr ) && "Wrong texture type" );
}

//-------------------------------------------------------------------------------------------------
Texture::Texture( 	UInt a_Width,
					UInt a_Height,
					UInt a_Depth,
					Format a_Format,
					Type a_Type,
					UInt a_MipMapCount )
: m_GLId( 0 )
, m_Width( a_Width )
, m_Height( a_Height )
, m_Depth( a_Depth )
, m_Format( a_Format )
, m_Type( a_Type )
, m_MipMapCount( a_MipMapCount )
{
	initOglInfos();

	Assert( ( m_Type == e2D || m_Type == e2Dr ) && "Wrong texture type" );
}

//-------------------------------------------------------------------------------------------------
void Texture::initOglInfos()
{
	switch ( m_Format )
	{
		case eBGRA:
		{
			m_OglFormat = GL_BGRA;
			m_OglInternalFormat = GL_RGBA;
			m_OglPixelDataType = GL_UNSIGNED_BYTE;
		}
		break;

		case eBGR:
		{
			m_OglFormat = GL_BGR;
			m_OglInternalFormat = GL_RGB;
			m_OglPixelDataType = GL_UNSIGNED_BYTE;
		}
		break;

		case eRGBA:
		{
			m_OglFormat = GL_RGBA;
			m_OglInternalFormat = GL_RGBA;
			m_OglPixelDataType = GL_UNSIGNED_BYTE;
		}
		break;

		case eRGB:
		{
			m_OglFormat = GL_RGB;
			m_OglInternalFormat = GL_RGB;
			m_OglPixelDataType = GL_UNSIGNED_BYTE;
		}
		break;

		case eRGBA16F:
		{
			m_OglFormat = GL_RGBA;
			m_OglInternalFormat = GL_RGBA16F;
			m_OglPixelDataType = GL_HALF_FLOAT;
		}
		break;

		case eRGB16F:
		{
			m_OglFormat = GL_RGB;
			m_OglInternalFormat = GL_RGB16F;
			m_OglPixelDataType = GL_HALF_FLOAT;
		}
		break;

		case eRGBA32F:
		{
			m_OglFormat = GL_RGBA;
			m_OglInternalFormat = GL_RGBA32F;
			m_OglPixelDataType = GL_FLOAT;
		}
		break;

		case eRGB32F:
		{
			m_OglFormat = GL_RGB;
			m_OglInternalFormat = GL_RGB32F;
			m_OglPixelDataType = GL_FLOAT;
		}
		break;

		case eDepth16:
		{
			m_OglFormat = GL_DEPTH_COMPONENT;
			m_OglInternalFormat = GL_DEPTH_COMPONENT16;
			m_OglPixelDataType = GL_UNSIGNED_INT;
		}
		break;

		case eDepth24:
		{
			m_OglFormat = GL_DEPTH_COMPONENT;
			m_OglInternalFormat = GL_DEPTH_COMPONENT24;
			m_OglPixelDataType = GL_UNSIGNED_INT;
		}
		break;

		case eDXTC1:
		{
			m_OglFormat = GL_COMPRESSED_RGBA_S3TC_DXT1_EXT;
		}
		break;

		case eDXTC3:
		{
			m_OglFormat = GL_COMPRESSED_RGBA_S3TC_DXT3_EXT;
		}
		break;

		case eDXTC5:
		{
			m_OglFormat = GL_COMPRESSED_RGBA_S3TC_DXT5_EXT;
		}
		break;

		case eL16F:
		{
			m_OglFormat = GL_LUMINANCE;
			m_OglInternalFormat = GL_LUMINANCE32F_ARB;
			m_OglPixelDataType = GL_FLOAT;
		}
		break;

		case eL32F:
		{
			m_OglFormat = GL_LUMINANCE;
			m_OglInternalFormat = GL_LUMINANCE32F_ARB;
			m_OglPixelDataType = GL_FLOAT;
		}
		break;

		default:
			Assert( 0 && "Texture::Texture - Unsupported Format" );
			break;
	}

	glGenTextures( 1, &m_GLId );
}

//-------------------------------------------------------------------------------------------------
Texture::~Texture()
{
	if ( m_GLId != 0 )
	{
		glDeleteTextures( 1, &m_GLId );
	}
}

//-------------------------------------------------------------------------------------------------
void Texture::init( const void * a_Buffer, bool a_ForceTexImage2D )
{
	glBindTexture( m_Type, m_GLId );
	glTexParameteri( m_Type, GL_TEXTURE_MAX_LEVEL, m_MipMapCount );

	if ( a_ForceTexImage2D || a_Buffer != NULL ) // a_Buffer can be null (RenderTarget case) but we still have to call glTexImage2D !!!
	{
		switch ( m_Type )
		{
			default:
				//Assert( 0 && "unknown texture type" );
				break;
			case e1D:
				glTexImage1D( m_Type, 0, m_OglInternalFormat, m_Width, 0, m_OglFormat, m_OglPixelDataType, a_Buffer );
				break;
			case e2D:
				glTexImage2D( m_Type, 0, m_OglInternalFormat, m_Width, m_Height, 0, m_OglFormat, m_OglPixelDataType, a_Buffer );
				break;
			case e3D:
				glTexImage3D( m_Type, 0, m_OglInternalFormat, m_Width, m_Height, m_Depth, 0, m_OglFormat, m_OglPixelDataType, a_Buffer );
				break;
		}
	}

	if ( m_Format == eDepth16 || m_Format == eDepth24 )
	{
		glTexParameteri( m_Type, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
		glTexParameteri( m_Type, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
		glTexParameteri( m_Type, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
		glTexParameteri( m_Type, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
		glTexParameteri( m_Type, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE );
		glTexParameteri( m_Type, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL );
	}
	else
	{
		if ( m_Format == eRGBA16F || m_Format == eRGB16F ||
			 m_Format == eRGBA32F || m_Format == eRGB32F ||
			 m_Format == eL16F || m_Format == eL32F )
		{
			glTexParameteri( m_Type, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
			glTexParameteri( m_Type, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
		}
		else
		{
			glTexParameteri( m_Type, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
			glTexParameteri( m_Type, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
		}

		glTexParameteri( m_Type, GL_TEXTURE_WRAP_S, GL_REPEAT );
		glTexParameteri( m_Type, GL_TEXTURE_WRAP_T, GL_REPEAT );
	}
}

//-------------------------------------------------------------------------------------------------
void Texture::use( unsigned int a_Idx ) const
{
	glActiveTexture( GL_TEXTURE0 + a_Idx );
    glBindTexture( m_Type, m_GLId );
    glEnable( m_Type );
}

void Texture::use( unsigned int a_Idx,
					 const Program * a_Program,
					 bool a_SetUniformTexSize,
					 int a_Location ) const
{
	Assert( a_Idx <= MaxTextureCount );

	int uLoc = ( a_Location == -1 ) ? ( a_Program->getUniformLocation( getTextureSamplersName( a_Idx ) ) ) : ( a_Location );
	// uLoc can be -1 if the shader doesn't use this textureSampler..
	if (uLoc == -1)
	{
		// LOG( DEBG, ( "Uniform [%s] could not be found [%d]", sTextureSamplers[ a_Idx ], a_Program->GetID() ) );
		return;
	}

	glActiveTexture( GL_TEXTURE0 + a_Idx );
    glBindTexture( m_Type, m_GLId );
    glEnable( m_Type );

    glUniform1i( uLoc, a_Idx );

    if ( a_SetUniformTexSize )
    {
    	float w = ( Float ) m_Width;
    	float h = ( Float ) m_Height;

    	const_cast<Program*>(a_Program)->setUniformVec4( getTextureSamplerSizesName( a_Idx ),
								   Vec4( w, h, 1 / w, 1 / h ) );
    }
}

//-------------------------------------------------------------------------------------------------
void Texture::release( unsigned int a_Idx ) const
{
	Assert( a_Idx <= MaxTextureCount );

	glActiveTexture( GL_TEXTURE0 + a_Idx );
    glBindTexture( m_Type, 0 );
    glDisable( m_Type );
}

//-------------------------------------------------------------------------------------------------
int Texture::computeSize( int nMipMapLevels )
{
	if ( isFormatCompressed() )
	{
		return sizeCompressed( m_Width >> nMipMapLevels, m_Height >> nMipMapLevels, m_Format);
	}
	else
	{
		return size( m_Width >> nMipMapLevels, m_Height >> nMipMapLevels, m_Format);
	}
}

//-------------------------------------------------------------------------------------------------
char * Texture::lock( 	int nMipMapLevels,
						int nCubeFace,
						char* & o_Buffer,
						char* & o_AlignedBuffer,
						int & o_LockedSize )
{
	if ( m_Type == e3D )
	{
		if ( nCubeFace == 0 )
		{
			o_LockedSize = computeSize( nMipMapLevels );

			o_Buffer = new char[ o_LockedSize * m_Depth + 128 ];
			o_AlignedBuffer = (char *) ( ( (Int64) o_Buffer + 127 ) & ~127 );
		}

		return &o_AlignedBuffer[ o_LockedSize * nCubeFace ];
	}
	else
	{
		o_LockedSize = computeSize( nMipMapLevels );

		o_Buffer = new char[ o_LockedSize + 128 ];
		o_AlignedBuffer = (char *) ( ( (Int64) o_Buffer + 127 ) & ~127 );

		return o_AlignedBuffer;
	}
}

//-------------------------------------------------------------------------------------------------
void Texture::unlock( 	int nMipMapLevels,
						int nCubeFace,
						char* & o_Buffer,
						char* & o_AlignedBuffer,
						int & o_LockedSize )
{
	if ( ( m_Type == e3D ) && ( nCubeFace != ( m_Depth - 1 ) ) ) return;

	glBindTexture( m_Type, m_GLId );

	GLint BindingTarget = m_Type;
	if ( m_Type == eCUBE )
	{
		BindingTarget = GL_TEXTURE_CUBE_MAP_POSITIVE_X + nCubeFace;
	}

	//glTexParameterf( GL_TEXTURE_2D, GL_GENERATE_MIPMAP, GL_TRUE);
	//glGenerateMipmap( GL_TEXTURE_2D);

	switch ( m_Type )
	{
		default:
			Assert( 0 && "unknown texture type" );

		case e3D:
		{
			if ( isFormatCompressed() )
			{
				glCompressedTexImage3DARB( BindingTarget, nMipMapLevels, m_OglFormat, m_Width >> nMipMapLevels, m_Height >> nMipMapLevels, m_Depth, 0,
						o_LockedSize, o_AlignedBuffer );
			}
			else
			{
				glTexImage3D( BindingTarget, nMipMapLevels, m_OglInternalFormat, m_Width >> nMipMapLevels, m_Height >> nMipMapLevels, m_Depth, 0,
						m_OglFormat, m_OglPixelDataType, o_AlignedBuffer );
			}
		}
		break;

		case e1D:
		{
			if ( isFormatCompressed() )
			{
				glCompressedTexImage1DARB( BindingTarget, nMipMapLevels, m_OglFormat, m_Width >> nMipMapLevels, 0, o_LockedSize, o_AlignedBuffer );
			}
			else
			{
				glTexImage1D( BindingTarget, nMipMapLevels, m_OglInternalFormat, m_Width >> nMipMapLevels, 0, m_OglFormat, m_OglPixelDataType,
						o_AlignedBuffer );
			}
		}
		break;

		case e2D:
		case eCUBE:
		{
			if ( isFormatCompressed() )
			{
				glCompressedTexImage2DARB( BindingTarget, nMipMapLevels, m_OglFormat, m_Width >> nMipMapLevels, m_Height >> nMipMapLevels, 0,
						o_LockedSize, o_AlignedBuffer );
			}
			else
			{
				glTexImage2D( BindingTarget, nMipMapLevels, m_OglInternalFormat, m_Width >> nMipMapLevels, m_Height >> nMipMapLevels, 0, m_OglFormat,
						m_OglPixelDataType, o_AlignedBuffer );
			}
		}
		break;
	}

	CHECK_GL_ERROR();

	delete[] o_Buffer;
	o_Buffer = NULL;
	o_AlignedBuffer = NULL;
	o_LockedSize = 0;
}


//-------------------------------------------------------------------------------------------------
UInt Texture::getId() const
{
	return m_GLId;
}

//-------------------------------------------------------------------------------------------------
const char *
Texture::getTextureSamplersName( UInt a_Unit )
{
	Assert( a_Unit < MaxTextureCount );
	return sTextureSamplers[ a_Unit ];
}

//-------------------------------------------------------------------------------------------------
const char *
Texture::getTextureSamplerSizesName( UInt a_Unit )
{
	Assert( a_Unit < MaxTextureCount );
	return sTextureSamplerSizes[ a_Unit ];
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
