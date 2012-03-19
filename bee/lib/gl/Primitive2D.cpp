/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: http://svn/drd/apps/bee/trunk/lib/gl/Primitive2D.cpp $"
 * SVN_META_ID = "$Id: Primitive2D.cpp 30989 2010-05-11 23:23:13Z chris.cooper $"
 */

#include "Primitive2D.h"
#include <GL/glew.h>
#include <GL/glut.h>
#include "Program.h"
#include "../kernel/spam.h"

using namespace bee;

Primitive2D::Primitive2D( Primitive2D::Type a_Type )
: m_Type( a_Type )
, m_PositionBufferID( 0 )
, m_TexCoordBufferID( 0 )
, m_ColorBufferID( 0 )
{
}

void Primitive2D::create( 	const void * a_PositionBuffer,
							const void * a_TexCoordBuffer,
							const void * a_ColorBuffer )
{
	// TODO: Sanity checks on these things
	if ( a_PositionBuffer != NULL )
	{
		glGenBuffers( 1, &m_PositionBufferID );
		glBindBuffer( GL_ARRAY_BUFFER, m_PositionBufferID );
		glBufferData( GL_ARRAY_BUFFER, getVertexCount() * 3 * sizeof(Float), a_PositionBuffer, GL_STATIC_DRAW );
	}

	if ( a_TexCoordBuffer != NULL )
	{
		glGenBuffers( 1, &m_TexCoordBufferID );
		glBindBuffer( GL_ARRAY_BUFFER, m_TexCoordBufferID );
		glBufferData( GL_ARRAY_BUFFER, getVertexCount() * 2 * sizeof(Float), a_TexCoordBuffer, GL_STATIC_DRAW );
	}

	if ( a_ColorBuffer != NULL )
	{
		glGenBuffers( 1, &m_ColorBufferID );
		glBindBuffer( GL_ARRAY_BUFFER, m_ColorBufferID );
		glBufferData( GL_ARRAY_BUFFER, getVertexCount() * 4 * sizeof(Float), a_ColorBuffer, GL_STATIC_DRAW );
	}
}

void Primitive2D::use( const Program * a_Program ) const
{
	bool normalize = false;

	if ( m_PositionBufferID != 0 )
	{
		glBindBuffer( GL_ARRAY_BUFFER, m_PositionBufferID );

		int attribLoc = a_Program->getAttribLocation( "iPosition" );
		BOOST_ASSERT( attribLoc != -1 && "Attribute(iVertex) not found !" );

		glVertexAttribPointer( attribLoc, 3, GL_FLOAT, normalize, 3 * sizeof(Float), 0 );
		glEnableVertexAttribArray( attribLoc );
	}

	if ( m_TexCoordBufferID != 0 )
	{
		glBindBuffer( GL_ARRAY_BUFFER, m_TexCoordBufferID );

		int attribLoc = a_Program->getAttribLocation( "iTexCoord" );
		if ( attribLoc != -1 ) // is that used by the program ?
		{
			glVertexAttribPointer( attribLoc, 2, GL_FLOAT, normalize, 2 * sizeof(Float), 0 );
			glEnableVertexAttribArray( attribLoc );
		}
	}

	if ( m_ColorBufferID != 0 )
	{
		glBindBuffer( GL_ARRAY_BUFFER, m_ColorBufferID );

		int attribLoc = a_Program->getAttribLocation( "iColor" );
		if ( attribLoc != -1 ) // is that used by the program ?
		{
			glVertexAttribPointer( attribLoc, 4, GL_FLOAT, //GL_BYTE, // use float colors for now..
					normalize, 4 * sizeof(Float), //sizeof(UChar),
					0 );
			glEnableVertexAttribArray( attribLoc );
		}
	}
}

void Primitive2D::draw()
{
	glDrawArrays( ( m_Type == eQuad ) ? ( GL_TRIANGLE_FAN ) : ( GL_TRIANGLES ), 0, getVertexCount() );

	// clean stuff
	glBindBuffer( GL_ARRAY_BUFFER, 0 );
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
