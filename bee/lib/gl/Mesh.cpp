/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: http://svn/drd/apps/bee/trunk/lib/gl/Mesh.cpp $"
 * SVN_META_ID = "$Id: Mesh.cpp 27186 2010-04-07 01:35:04Z david.morris $"
 */

#include "Mesh.h"
#include "Program.h"
#include "../kernel/spam.h"
#include <boost/assert.hpp>

using namespace bee;

Mesh::Mesh( unsigned int a_VertexDataCount )
: m_VertexBufferID( 0 )
, m_ColourBufferID( 0 )
, m_NormalBufferID( 0 )
, m_TexCoordBufferID( 0 )
, m_VertexDataCount( a_VertexDataCount )
, m_VertexElementCount( 0 )
, m_VertexElementSize( 0 )
, m_ColourElementCount( 0 )
, m_ColourElementSize( 0 )
, m_NormalElementCount( 0 )
, m_NormalElementSize( 0 )
, m_TexCoordElementCount( 0 )
, m_TexCoordElementSize( 0 )
{
}

Mesh::~Mesh()
{
	if ( m_VertexBufferID > 0 ) glDeleteBuffers( 1, &m_VertexBufferID );
	if ( m_ColourBufferID > 0 ) glDeleteBuffers( 1, &m_ColourBufferID );
	if ( m_NormalBufferID > 0 ) glDeleteBuffers( 1, &m_NormalBufferID );
	if ( m_TexCoordBufferID > 0 ) glDeleteBuffers( 1, &m_TexCoordBufferID );
}

void Mesh::createVertexBuffer( 	unsigned int a_ElementCount,
								unsigned int a_ElementSize,
								void * a_Buffer )
{
	glGenBuffers( 1, &m_VertexBufferID );
	glBindBuffer( GL_ARRAY_BUFFER, m_VertexBufferID );
	glBufferData( GL_ARRAY_BUFFER, getVertexDataCount() * a_ElementCount * a_ElementSize, a_Buffer, GL_STATIC_DRAW );

	m_VertexElementCount = a_ElementCount;
	m_VertexElementSize = a_ElementSize;
}

void Mesh::createNormalBuffer( 	unsigned int a_ElementCount,
								unsigned int a_ElementSize,
								void * a_Buffer )
{
	glGenBuffers( 1, &m_NormalBufferID );
	glBindBuffer( GL_ARRAY_BUFFER, m_NormalBufferID );
	glBufferData( GL_ARRAY_BUFFER, getVertexDataCount() * a_ElementCount * a_ElementSize, a_Buffer, GL_STATIC_DRAW );

	m_NormalElementCount = a_ElementCount;
	m_NormalElementSize = a_ElementSize;
}

void Mesh::createColourBuffer( 	unsigned int a_ElementCount,
								unsigned int a_ElementSize,
								void * a_Buffer )
{
	glGenBuffers( 1, &m_ColourBufferID );
	glBindBuffer( GL_ARRAY_BUFFER, m_ColourBufferID );
	glBufferData( GL_ARRAY_BUFFER, getVertexDataCount() * a_ElementCount * a_ElementSize, a_Buffer, GL_STATIC_DRAW );

	m_ColourElementCount = a_ElementCount;
	m_ColourElementSize = a_ElementSize;
}

void Mesh::createTexCoordBuffer( 	unsigned int a_ElementCount,
									unsigned int a_ElementSize,
									void * a_Buffer )
{
	glGenBuffers( 1, &m_TexCoordBufferID );
	glBindBuffer( GL_ARRAY_BUFFER, m_TexCoordBufferID );
	glBufferData( GL_ARRAY_BUFFER, getVertexDataCount() * a_ElementCount * a_ElementSize, a_Buffer, GL_STATIC_DRAW );

	m_TexCoordElementCount = a_ElementCount;
	m_TexCoordElementSize = a_ElementSize;
}

#define USE_VERTEX_ATTRIB_POINTER
void Mesh::use( const Program * a_Program ) const
{
	bool normalize = false;

	// Vertex
	glBindBuffer( GL_ARRAY_BUFFER, m_VertexBufferID );
#ifdef USE_VERTEX_ATTRIB_POINTER
	int attribLoc = a_Program->getAttribLocation( "iPosition" );
	BOOST_ASSERT( attribLoc != -1 && "Attribute(iVertex) not found !" );

	glVertexAttribPointer( attribLoc,
						   getVertexElementCount(),
						   GL_FLOAT,
						   normalize,
						   getVertexElementCount() * getVertexElementSize(),
						   0 );
	glEnableVertexAttribArray(attribLoc);
#else
	glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer( GetVertexElementCount(), GL_FLOAT, 0, NULL );
#endif

	// Colour (if we have any)
	if ( m_ColourBufferID > 0 )
	{
		glBindBuffer( GL_ARRAY_BUFFER, m_ColourBufferID );
#ifdef USE_VERTEX_ATTRIB_POINTER
		int attribLoc = a_Program->getAttribLocation( "iColour" );
		if ( attribLoc != -1 ) // is that used by the program ?
		{
			glVertexAttribPointer( attribLoc,
								   getColourElementCount(),
								   GL_FLOAT,
								   normalize,
								   getColourElementCount() * getColourElementSize(),
								   0 );
			glEnableVertexAttribArray(attribLoc);
		}
#else
		glEnableClientState(GL_COLOR_ARRAY);
		glNormalPointer( GL_FLOAT, 0, NULL );
#endif
	}

	// Normal (if we have some)
	if ( m_NormalBufferID > 0 )
	{
		glBindBuffer( GL_ARRAY_BUFFER, m_NormalBufferID );
#ifdef USE_VERTEX_ATTRIB_POINTER
		int attribLoc = a_Program->getAttribLocation( "iNormal" );
		if ( attribLoc != -1 ) // is that used by the program ?
		{
			glVertexAttribPointer( attribLoc,
								   getNormalElementCount(),
								   GL_FLOAT,
								   normalize,
								   getNormalElementCount() * getNormalElementSize(),
								   0 );
			glEnableVertexAttribArray(attribLoc);
		}
#else
		glEnableClientState(GL_NORMAL_ARRAY);
		glNormalPointer( GL_FLOAT, 0, NULL );
#endif
	}

	// TexCoord (if we have some)
	if ( m_TexCoordBufferID > 0 ) // todo rename in UV0 and manage more than one UV..
	{
		glBindBuffer( GL_ARRAY_BUFFER, m_TexCoordBufferID );
#ifdef USE_VERTEX_ATTRIB_POINTER
		int attribLoc = a_Program->getAttribLocation( "iTexCoord" );
		if ( attribLoc != -1 ) // is that used by the program ?
		{
			glVertexAttribPointer( attribLoc,
					getTexCoordElementCount(),
					GL_FLOAT,
					normalize,
					getTexCoordElementCount() * getTexCoordElementSize(),
					0 );
			glEnableVertexAttribArray(attribLoc);
		}
#else
		glEnableClientState(GL_TEXTURE_COORD_ARRAY);
		glTexCoordPointer( getTexCoordElementCount(), GL_FLOAT, 0, NULL );
#endif
	}
}

void Mesh::draw( UInt a_Type )
{
	//SPAM(GetVertexDataCount());
	glDrawArrays(a_Type, 0, getVertexDataCount() );

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
