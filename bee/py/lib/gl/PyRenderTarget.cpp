/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: http://svn/drd/apps/bee/trunk/py/lib/gl/PyRenderTarget.cpp $"
 * SVN_META_ID = "$Id: PyRenderTarget.cpp 26502 2010-03-26 05:12:33Z david.morris $"
 */

//----------------------------------------------------------------------------
// system includes
#include <stdexcept>
// bee includes
#include <gl/Program.h>
#include <gl/RenderTarget.h>
// bee::py includes
#include "PyRenderTarget.h"
#include "PyTexture.h"

//----------------------------------------------------------------------------
using namespace bee::py;

//----------------------------------------------------------------------------
RenderTarget::RenderTarget()
{
}

//----------------------------------------------------------------------------
RenderTarget::RenderTarget( bee::UInt a_width, bee::UInt a_height )
{
	create( a_width, a_height );
}

//----------------------------------------------------------------------------
RenderTarget::~RenderTarget()
{
	destroy();
}

//----------------------------------------------------------------------------
bool
RenderTarget::create( bee::UInt a_width, bee::UInt a_height )
{
	try
	{
		m_renderTarget.reset( new bee::RenderTarget( a_width,
		                                             a_height,
		                                             bee::Texture::eRGBA,
		                                             bee::Texture::e2D,
		                                             1,
		                                             bee::Texture::eDepth24 ) );
		if ( m_renderTarget ) return true;
	}
	catch ( ... )
	{
	}

	throw std::runtime_error( std::string( "error creating render target" ) );
	return false;
}

//----------------------------------------------------------------------------
bool
RenderTarget::destroy()
{
	m_renderTarget.reset();
}

//----------------------------------------------------------------------------
void
RenderTarget::use()
{
	m_renderTarget->use();
}

//----------------------------------------------------------------------------
void
RenderTarget::release()
{
	m_renderTarget->release();
}

//----------------------------------------------------------------------------
const bee::py::Texture *
RenderTarget::getTexture( bee::UInt a_Unit ) const
{
	return new bee::py::Texture( m_renderTarget->getTexture( a_Unit ) );
}

//----------------------------------------------------------------------------
bee::UInt
RenderTarget::getWidth() const
{
	return m_renderTarget->getWidth();
}

//----------------------------------------------------------------------------
bee::UInt
RenderTarget::getHeight() const
{
	return m_renderTarget->getHeight();
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
