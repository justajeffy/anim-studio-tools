/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: http://svn/drd/apps/bee/trunk/lib/gl/RenderTarget.cpp $"
 * SVN_META_ID = "$Id: RenderTarget.cpp 30989 2010-05-11 23:23:13Z chris.cooper $"
 */

#include "RenderTarget.h"
#include "glError.h"
#include <GL/glew.h>
#include <GL/glut.h>
#include "../kernel/spam.h"
#include <iostream>
#include <sstream>

using namespace bee;

void RenderTarget::checkSupportGLContext()
{
	Int maxbuffers;
	glGetIntegerv( GL_MAX_COLOR_ATTACHMENTS, &maxbuffers );

	Assert( m_Count <= maxbuffers );
}

RenderTarget::RenderTarget( UInt a_Width,
							UInt a_Height )
: m_Count( 1 )
, m_Surface( 0 )
, m_Format( Texture::eNone )
, m_DepthFormat( Texture::eNone )
, m_TextureArray( NULL )
, m_DepthTexture( NULL )
{
	// utility constructor for default rendertarget (the screen)
}

RenderTarget::RenderTarget( UInt a_Width,
							UInt a_Height,
							Texture::Format a_Format,
							Texture::Type a_Type,
							UInt a_Count,
							Texture::Format a_DepthFormat )
: m_Count( a_Count )
, m_Format( a_Format )
, m_Type( a_Type )
, m_DepthFormat( a_DepthFormat )
, m_Width( a_Width )
, m_Height( a_Height )
, m_DepthTexture( NULL )
{
	Assert( a_Count > 0 && "RenderTarget::RenderTarget - Count parameter can not be null !" );

	checkSupportGLContext();

	// create surface
	glGenFramebuffers( 1, &m_Surface );
	glBindFramebuffer( GL_FRAMEBUFFER, m_Surface );
	CHECK_GL_ERROR();

	if ( m_Format != Texture::eNone )
	{
		Assert( m_Count > 0 && "Texture count is null !");
		m_TextureArray = new Texture*[ m_Count ];

		for ( UInt iTex = 0 ; iTex < m_Count ; ++iTex )
		{
			m_TextureArray[ iTex ] = new Texture( a_Width, a_Height, a_Format, a_Type );
			m_TextureArray[ iTex ]->init( NULL, true );
			CHECK_GL_ERROR();

			// attach texture to framebuffer object
			glFramebufferTexture2D( GL_FRAMEBUFFER, iTex + GL_COLOR_ATTACHMENT0, m_TextureArray[ iTex ]->m_Type, m_TextureArray[ iTex ]->m_GLId, 0 );
			CHECK_GL_ERROR();
		}
	}

	if ( m_DepthFormat != Texture::eNone )
	{
		m_DepthTexture = new Texture( a_Width, a_Height, a_DepthFormat, a_Type );
		m_DepthTexture->init( NULL, true );

		glFramebufferTexture2D( GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, m_DepthTexture->m_GLId, 0 );
	}

	int nError = glCheckFramebufferStatus( GL_FRAMEBUFFER );
	Assert( nError == GL_FRAMEBUFFER_COMPLETE && "RenderTarget was not created successfully." );

	restore();
}

RenderTarget::~RenderTarget()
{
	if ( m_Format != Texture::eNone )
	{
		for (UInt iTex = 0; iTex < m_Count; ++iTex)
		{
			delete m_TextureArray[ iTex ];
		}

		delete [] m_TextureArray;
	}

	if ( m_DepthFormat != Texture::eNone )
	{
		delete m_DepthTexture;
	}

	// delete surface
	glDeleteFramebuffers( 1, &m_Surface );
}

void RenderTarget::resize( 	UInt a_Width,
							UInt a_Height )
{
	Assert( m_Surface == 0 && "RenderTarget::Resize - can only resize default rendertarget !" );
	m_Width = a_Width;
	m_Height = a_Height;
}

void RenderTarget::restore()
{
	glBindFramebuffer( GL_FRAMEBUFFER, 0 );
	glBindRenderbuffer( GL_RENDERBUFFER, 0 );
	glDrawBuffer( GL_BACK );
	glReadBuffer( GL_BACK );
}

void RenderTarget::use()
{
	if ( m_Surface == 0 )
	{
		restore();
		glViewport( 0, 0, m_Width, m_Height );
	}
	else
	{
		glBindFramebuffer( GL_FRAMEBUFFER, m_Surface );
		CHECK_GL_ERROR();

		glViewport( 0, 0, m_Width, m_Height );

		if ( isDepthOnly() )
		{
			glDrawBuffer(GL_NONE);
			glReadBuffer(GL_NONE);
		}
		else if ( m_Count > 1 )
		{
			static const GLenum buffers[] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2, GL_COLOR_ATTACHMENT3 };
			glDrawBuffers( m_Count, buffers );
			CHECK_GL_ERROR();
		}
		else
		{
			glReadBuffer( GL_COLOR_ATTACHMENT0_EXT );
		}
	}
}

void RenderTarget::release()
{
	if ( m_Surface != 0 )
		restore();
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
