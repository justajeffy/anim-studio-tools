/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: http://svn.drd.int/drd/apps/bee/trunk/lib/io/objLoader.h $"
 * SVN_META_ID = "$Id: ptcTree.cpp 39194 2010-07-15 23:40:45Z stephane.bertout $"
 */

#include "io/ptcTree.h"
#include "io/ptcNode.h"

using namespace bee;

PtcTree * PtcTree::s_Instance = NULL;

PtcTree::PtcTree( int a, int s )
: m_ElementAllocated( a )
, m_ElementCount( 0 )
, m_ElementSize( s )
, m_PtcCellBuffer( NULL )
, m_PtcCellCount( 0 )
, m_PtcCellAllocated( 0 )
, m_PtcNodeBuffer( NULL )
, m_PtcNodeCount( 0 )
, m_PtcNodeAllocated( 0 )
{
	Init();
}

PtcTree::PtcTree( const std::string & a_FileName)
: m_ElementAllocated( 0 )
, m_ElementCount( 0 )
, m_ElementSize( 0 )
, m_PtcCellBuffer( NULL )
, m_PtcCellCount( 0 )
, m_PtcCellAllocated( 0 )
, m_PtcNodeBuffer( NULL )
, m_PtcNodeCount( 0 )
, m_PtcNodeAllocated( 0 )
{
	LoadFromFile( a_FileName );
}

PtcTree::PtcTree( const PtcTree & a, void * a_ElementBuffer, void * a_PtcCellBuffer, void * a_PtcNodeBuffer )
: m_ElementAllocated( a.m_ElementAllocated )
, m_ElementCount( a.m_ElementCount )
, m_ElementSize( a.m_ElementSize )
, m_ElementBuffer( (char *) a_ElementBuffer )
, m_PtcCellBuffer( (PtcCell*) a_PtcCellBuffer )
, m_PtcCellCount( a.m_PtcCellCount )
, m_PtcCellAllocated( a.m_PtcCellAllocated )
, m_PtcNodeBuffer( (PtcNode*) a_PtcNodeBuffer )
, m_PtcNodeCount( a.m_PtcNodeCount )
, m_PtcNodeAllocated( a.m_PtcNodeAllocated )
{
}

void PtcTree::Init()
{
	m_ElementBuffer = new char [ m_ElementAllocated * m_ElementSize ];

	Assert( s_Instance == NULL );
	s_Instance = this;
}

void PtcTree::MoveAllocatedBuffer() // dbg
{
	char * newElementBuffer = new char [ m_ElementAllocated * m_ElementSize ];
	memcpy( newElementBuffer, m_ElementBuffer, m_ElementAllocated * m_ElementSize );
	delete [] m_ElementBuffer;
	m_ElementBuffer = newElementBuffer;
}

PtcTree::~PtcTree()
{
	delete [] ((char *) m_ElementBuffer);
	if ( s_Instance == this) s_Instance = NULL;
}

void PtcTree::Allocate( int count_needed, char * & o_ElementBuffer )
{
	assert( ( m_ElementCount + count_needed) <= m_ElementAllocated ); // as we reserve enough in the pool this should never assert !

	o_ElementBuffer = m_ElementBuffer + m_ElementCount * m_ElementSize;
	m_ElementCount += count_needed;
}

void PtcTree::SaveToFile( const std::string & a_FileName )
{
	FILE * newFile = fopen( a_FileName.c_str(), "w" );

	fwrite( "OCD ", 1, 4, newFile );
	fwrite( &m_ElementAllocated, 1, sizeof(int), newFile );
	fwrite( &m_ElementSize, 1, sizeof(int), newFile );
	fwrite( m_ElementBuffer, 1, m_ElementAllocated * m_ElementSize, newFile );

	Assert( m_PtcCellCount ==  m_PtcCellAllocated );
	fwrite( &m_PtcCellAllocated, 1, sizeof(int), newFile );
	fwrite( m_PtcCellBuffer, 1, m_PtcCellAllocated * sizeof(PtcCell), newFile );

	Assert( m_PtcNodeCount ==  m_PtcNodeAllocated );
	fwrite( &m_PtcNodeAllocated, 1, sizeof(int), newFile );
	fwrite( m_PtcNodeBuffer, 1, m_PtcNodeAllocated * sizeof(PtcNode), newFile );

	fclose( newFile );
}

void PtcTree::LoadFromFile( const std::string & a_FileName )
{
	FILE * file = fopen( a_FileName.c_str(), "r" );

	char tag[4];
	fread( tag, 1, 4, file );
	Assert( tag[0] == 'O' && tag[1] == 'C' && tag[2] == 'D' && tag[3] == ' ' );

	fread( &m_ElementAllocated, 1, sizeof(int), file );
	fread( &m_ElementSize, 1, sizeof(int), file );
	Init();

	fread( m_ElementBuffer, 1, m_ElementAllocated * m_ElementSize, file );

	Assert( m_PtcCellCount == 0 && m_PtcCellAllocated == 0 && m_PtcCellBuffer == NULL );
	int m_PtcCellAllocated;
	fread( &m_PtcCellAllocated, 1, sizeof(int), file );
	AllocatePtcCellMemPool( m_PtcCellAllocated );
	fread( m_PtcCellBuffer, 1, m_PtcCellAllocated * sizeof(PtcCell), file );

	Assert( m_PtcNodeCount == 0 && m_PtcNodeAllocated == 0 && m_PtcNodeBuffer == NULL );
	int m_PtcNodeAllocated;
	fread( &m_PtcNodeAllocated, 1, sizeof(int), file );
	AllocatePtcNodeMemPool( m_PtcNodeAllocated );
	fread( m_PtcNodeBuffer, 1, m_PtcNodeAllocated * sizeof(PtcNode), file );

	// all is full
	m_PtcCellCount = m_PtcCellAllocated;
	m_PtcNodeCount = m_PtcNodeAllocated;
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
