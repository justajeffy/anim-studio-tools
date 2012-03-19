/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: http://svn/drd/apps/bee/trunk/lib/gl/PBO.cpp $"
 * SVN_META_ID = "$Id: PBO.cpp 27186 2010-04-07 01:35:04Z david.morris $"
 */

#include "PBO.h"
#include "glExtensions.h"
#include <stdio.h>

using namespace bee;

//-------------------------------------------------------------------------------------------------
PBO::PBO( UInt a_Width, UInt a_Height, const void * a_Buffer, UInt a_BufferSize )
:	m_BufGLId( 0 )
,	m_Width( a_Width )
,	m_Height( a_Height )
,	m_Buffer( a_Buffer )
,	m_BufferSize( a_BufferSize )
{
	glGenBuffers( 1, &m_BufGLId );
	glBindBuffer( GL_PIXEL_UNPACK_BUFFER, m_BufGLId );
	glBufferData( GL_PIXEL_UNPACK_BUFFER, m_BufferSize, m_Buffer, GL_STREAM_DRAW );
}

//-------------------------------------------------------------------------------------------------
PBO::~PBO()
{
	if ( m_BufGLId != 0 )
	{
		glDeleteBuffers( 1, &m_BufGLId );
	}
}


//-------------------------------------------------------------------------------------------------
void PBO::use() const
{
	glBindBuffer( GL_PIXEL_UNPACK_BUFFER, m_BufGLId );
}

//-------------------------------------------------------------------------------------------------
void PBO::release() const
{
	glBindBuffer( GL_PIXEL_UNPACK_BUFFER, 0 );
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
