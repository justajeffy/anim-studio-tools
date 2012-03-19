/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: http://svn/drd/apps/bee/trunk/lib/gl/GenericMesh.cpp $"
 * SVN_META_ID = "$Id: GenericMesh.cpp 30989 2010-05-11 23:23:13Z chris.cooper $"
 */

#include <GL/glew.h>
#include <GL/glut.h>
#include "GenericMesh.h"
#include "Program.h"
#include "streams.h"
#include "../kernel/log.h"
#include <boost/assert.hpp>

using namespace bee;

//-------------------------------------------------------------------------------------------------
GenericMesh::GenericMesh( 	UInt a_NumberOfVertexStreams,
							Bool a_HasIndexStream )
:	m_VertexStreams( NULL )
,	m_IndexStream( NULL )
,	m_NumberOfVertexStreams( a_NumberOfVertexStreams )
,	m_HasIndexStream( a_HasIndexStream )
{
	m_VertexStreams = new VertexStream*[ m_NumberOfVertexStreams ];
	for ( UInt idx = 0 ; idx < m_NumberOfVertexStreams ; ++idx )
		m_VertexStreams[ idx ] = NULL;
}

//-------------------------------------------------------------------------------------------------
GenericMesh::~GenericMesh()
{
	for ( UInt idx = 0 ; idx < m_NumberOfVertexStreams ; ++idx )
	{
		delete m_VertexStreams[ idx ];
		m_VertexStreams[ idx ] = NULL;
	}

	delete [] m_VertexStreams;
	delete m_IndexStream;

	m_VertexStreams = NULL;
	m_IndexStream = NULL;
}

//-------------------------------------------------------------------------------------------------
UInt
GenericMesh::addStream( VertexStream * a_Stream )
{
	ASSERT( a_Stream );

	UInt idx = 0;
	for ( ; idx < m_NumberOfVertexStreams && m_VertexStreams[ idx ] ; ++idx )
		;
	ASSERT( idx < m_NumberOfVertexStreams && "Trying to add too many streams, no twinkie!" );
//	LOG( DEBG, ( "Adding stream #%d", idx ) );
	m_VertexStreams[ idx ] = a_Stream;

	UInt glId = 0;
	glGenBuffers( 1, &glId );
	ASSERT( glId != 0 && "Invalid glID!" );

	glBindBuffer( GL_ARRAY_BUFFER, glId );
	glBufferData( GL_ARRAY_BUFFER, a_Stream->getSize(), a_Stream->getData(), GL_STATIC_DRAW );
	a_Stream->setGLId( glId );
	glBindBuffer( GL_ARRAY_BUFFER, 0 );

	// update the BBox:
	if ( idx == 0 )
	{
		Vec3 * verticesH = ( Vec3 * ) a_Stream->getData();
		for ( UInt idx = 0 ; idx < a_Stream->getNumElements() ; ++idx )
		{
			m_BBox.update( verticesH[ idx ] );
		}
//		const Vec3 mi = m_BBox.Min();
//		const Vec3 ma = m_BBox.Max();
//		LOG( DEBG, ( "min: %9.5f, %9.5f, %9.5f max: %9.5f, %9.5f, %9.5f", mi.x, mi.y, mi.z, ma.x, ma.y, ma.z ));
	}

	return idx;
}

//-------------------------------------------------------------------------------------------------
Bool
GenericMesh::addStream( IndexStream * a_Stream )
{
	ASSERT( m_HasIndexStream && "Can't add IndexStream if you already told me that I don't have one!" );
	ASSERT( a_Stream );
	ASSERT( !m_IndexStream && "Already have an IndexStream!" );

	m_IndexStream = a_Stream;

	UInt glId = 0;
	glGenBuffers( 1, &glId );
	ASSERT( glId != 0 && "Invalid glID!" );

	glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, glId );
	glBufferData( GL_ELEMENT_ARRAY_BUFFER, a_Stream->getSize(), a_Stream->getData(), GL_STATIC_DRAW );
	a_Stream->setGLId( glId );
	glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, 0 );
	return true;
}


//-------------------------------------------------------------------------------------------------
void
GenericMesh::use( const Program * a_Program ) const
{
	for ( UInt idx = 0 ; idx < m_NumberOfVertexStreams ; ++idx )
	{
		VertexStream * vs = m_VertexStreams[ idx ];
		if ( !m_VertexStreams[ idx ] )
			continue;

		glBindBuffer( GL_ARRAY_BUFFER, vs->getGLId() );
		int attribLoc = a_Program->getAttribLocation( vs->getName() );
		if ( attribLoc == -1 )
		{
			// don't spam too much!
			// std::cerr << "Didn't find attrib [" << vs->GetName() << "]" << std::endl;
			continue;
		}
		glVertexAttribPointer( attribLoc,
							   vs->getNumComponentsPerElement(),
							   vs->getGLComponentType(),
							   false,
							   vs->getStride(),
							   0 );
		glEnableVertexAttribArray(attribLoc);
	}

	if ( m_HasIndexStream )
	{
		ASSERT( m_IndexStream && "What? How can we have indexStream without an indexStream!?" );
		glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, m_IndexStream->getGLId() );
	}
}

//-------------------------------------------------------------------------------------------------
void
GenericMesh::release( const Program * a_Program ) const
{
	for ( UInt idx = 0 ; idx < m_NumberOfVertexStreams ; ++idx )
	{
		VertexStream * vs = m_VertexStreams[ idx ];
		if ( !m_VertexStreams[ idx ] )
			continue;

		glBindBuffer( GL_ARRAY_BUFFER, vs->getGLId() );
		int attribLoc = a_Program->getAttribLocation( vs->getName() );
		if ( attribLoc == -1 )
		{
			// don't spam too much!
			// std::cerr << "Didn't find attrib [" << vs->GetName() << "]" << std::endl;
			continue;
		}
		glDisableVertexAttribArray(attribLoc);
		glBindBuffer( GL_ARRAY_BUFFER, 0 );
	}

	if ( m_HasIndexStream )
	{
		ASSERT( m_IndexStream && "What? How can we have indexStream without an indexStream!?" );
		glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, 0 );
	}
}

//-------------------------------------------------------------------------------------------------
void
GenericMesh::draw() const
{
	ASSERT( m_HasIndexStream );

	glDrawElements( m_IndexStream->getGLDrawType(),
					m_IndexStream->getNumElements() * m_IndexStream->getNumComponentsPerElement(),
					m_IndexStream->getGLComponentType(),
					0 );
	glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, 0 );
}

//-------------------------------------------------------------------------------------------------
void
GenericMesh::draw( UInt a_DrawType ) const
{
	ASSERT( m_NumberOfVertexStreams > 0 && m_VertexStreams[ 0 ] != NULL );
	VertexStream * vs = m_VertexStreams[ 0 ];
	glDrawArrays( a_DrawType, 0, vs->getNumElements() );
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
