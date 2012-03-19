/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: http://svn/drd/apps/bee/trunk/lib/gl/RenderTarget.h $"
 * SVN_META_ID = "$Id: RenderTarget.h 27186 2010-04-07 01:35:04Z david.morris $"
 */

#ifndef bee_RenderTarget_h
#define bee_RenderTarget_h
#pragma once

#include "Texture.h"
#include "../kernel/types.h"
#include "../kernel/assert.h"

namespace bee
{
	//-------------------------------------------------------------------------------------------------
	//! RenderTarget is a GL utility class allowing manipulation of GL RenderTarget
	class RenderTarget
	{
	public:
		//! Default Constructor (usually the GL context basic framebuffer.. used to retain its size)
		RenderTarget( 	UInt a_Width,
						UInt a_Height );
		//! Constructor (Count contains the number of Colour buffer to create, use eNone as DepthFormat if you don't need a depth buffer)
		RenderTarget( 	UInt a_Width,
						UInt a_Height,
						Texture::Format a_Format,
						Texture::Type a_Type,
						UInt a_Count,
						Texture::Format a_DepthFormat );
		//! Destructor
		~RenderTarget();

		//! Setup the GL State
		void use();
		void release();

		//! Return True if the RenderTarget contains a DepthBuffer
		inline bool containsDepthBuffer() const
		{
			return m_DepthFormat != Texture::eNone;
		}
		//! Return True if the RenderTarget <b>only</b> contains a DepthBuffer
		inline bool isDepthOnly() const
		{
			return containsDepthBuffer() && ( getFormat() == Texture::eNone );
		}

		//! Returns a texture associated to the RenderTarget
		const Texture * getTexture( UInt a_Idx = 0 )
		{
			Assert( m_TextureArray != NULL );
			Assert( a_Idx < m_Count );
			return m_TextureArray[ a_Idx ];
		}
		//! Returns a depth texture associated to the RenderTarget
		const Texture * getDepthTexture()
		{
			Assert( containsDepthBuffer() );
			Assert( m_DepthTexture != NULL );
			return m_DepthTexture;
		}

		//! Function only available for default RenderTarget (see Default Constructor) - this will assert if used with a normal RenderTarget
		void resize( UInt a_Width, UInt a_Height );

		//! Returns the number of colour textures created
		inline UInt getCount() const
		{
			return m_Count;
		}
		//! Returns the colour texture format
		inline Texture::Format getFormat() const
		{
			return m_Format;
		}

		//! get width
		UInt getWidth() const { return m_Width; }

		//! get height
		UInt getHeight() const { return m_Height; }

	private:
		void checkSupportGLContext();
		void restore();

		UInt				m_Count;
		UInt				m_Surface;
		Texture::Format		m_Format;
		Texture::Type		m_Type;
		Texture::Format		m_DepthFormat;
		UInt 				m_Width;
		UInt 				m_Height;

		Texture**			m_TextureArray;
		Texture* 			m_DepthTexture;
	};
}

#endif // bee_RenderTarget_h


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
