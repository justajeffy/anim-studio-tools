/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: http://svn.drd.int/drd/apps/bee/trunk/lib/io/objLoader.h $"
 * SVN_META_ID = "$Id: ptcTreeBuilder.cpp 41740 2010-08-08 23:21:56Z allan.johns $"
 */

#include "io/ptcTreeBuilder.h"
#include "io/ptcNode.h"
#include "io/ptcTree.h"
#include "math/Imath.h"

using namespace bee;

PtcTreeBuilder * PtcTreeBuilder::s_Instance = NULL;

PtcTreeBuilder::PtcTreeBuilder( int a_SplitCount, const Imath::V3f & a_Min, const Imath::V3f & a_Max  )
: m_SplitCount( a_SplitCount )
, m_Min( a_Min )
, m_Max( a_Max )
{
	m_PtcCells = new PtcCell[ m_SplitCount * m_SplitCount * m_SplitCount ];

	Vec3 range = m_Max - m_Min;
	m_SplitRange = range / ((float) m_SplitCount);

	Assert( s_Instance == NULL );
	s_Instance = this;
}

PtcTreeBuilder::~PtcTreeBuilder()
{
	FreeMemory();

	if ( s_Instance == this) s_Instance = NULL;
}

const PtcTree * PtcTreeBuilder::BuildTree( int a_PointCount, std::vector< float > & vertexVec, std::vector< float > & normalVec, std::vector< float > & radiusVec )
{
	printf( "min = %f %f %f \n", GetMin().x, GetMin().y, GetMin().z );
	printf( "max = %f %f %f \n", GetMax().x, GetMax().y, GetMax().z );
	printf( "splitRange = %f %f %f \n", GetSplitRange().x, GetSplitRange().y, GetSplitRange().z );

	printf( "getPointCount() = %d \n", a_PointCount );

	for ( int i = 0; i < a_PointCount; ++i )
	{
		Vec3 position( vertexVec[i*3], vertexVec[i*3+1], vertexVec[i*3+2] );
		Vec3 normal( normalVec[i*3], normalVec[i*3+1], normalVec[i*3+2] );

		int xPos, yPos, zPos;
		GetPtcCellsIndexers( position, xPos, yPos, zPos );

		PtcCell * ptcs = GetPtcCell(xPos,yPos,zPos);

		ptcs->SetBox( xPos, yPos, zPos, GetSplitRange(), GetMin() );
		ptcs->Add( position, normal, radiusVec[i] );
	}

	// we can now free the vectors..
	vertexVec.clear();
	normalVec.clear();
	radiusVec.clear();

	// how many valid cells do we have ?
	int total = 0, valid_count = 0;
	for(int iSplitX = 0; iSplitX < m_SplitCount; ++iSplitX)
	{
		for(int iSplitY = 0; iSplitY < m_SplitCount; ++iSplitY)
		{
			for(int iSplitZ = 0; iSplitZ < m_SplitCount; ++iSplitZ)
			{
				PtcCell * ptcCell = GetPtcCell( iSplitX, iSplitY, iSplitZ );
				total += ptcCell->GetCount();

				if ( ptcCell->GetCount() > 0 ) valid_count ++;
			}
		}
	}

	printf("total = %d | valid = %d | avg point count per cell = %d /// %d (%d)\n", total, valid_count, a_PointCount / valid_count, m_SplitCount * m_SplitCount * m_SplitCount, m_SplitCount );
	assert( total == a_PointCount );

	printf("Starting populating RootNode... \n" );

	PtcNode * rootNode = new PtcNode( GetMin(), GetMax(), 0, 0, 0, m_SplitCount, m_SplitCount, m_SplitCount );
	rootNode->Populate();

	assert( rootNode->GetPointCount() == a_PointCount );

	// now let's repack the asset in one big memory buffer !
	PtcTree * tree = new PtcTree( a_PointCount, SIZEOF_ELEMENT ); // position & normal

	tree->AllocatePtcCellMemPool( valid_count );
	rootNode->RepackAssets();
	assert( tree->IsEmpty() == true );
	assert( tree->IsPtcCellMemPoolEmpty() == true );

	// we can now delete the huge array of PtcCell
	FreeMemory();

	// now let's repack all the ptc nodes in one mempool
	PtcTree::Instance()->AllocatePtcNodeMemPool( PtcNode::ptcNodeCreatedCount );
	PtcNode * newRootNode = tree->AllocatePtcNode();
	rootNode->RepackTo( newRootNode );
	delete rootNode;
	rootNode = newRootNode;

	assert( tree->IsPtcNodeMemPoolEmpty() == true );

	//printf( "sizeof PtcNode Mem Pool = %s (%d elements) \n", getReadableSize( tree->GetPtcNodeCount() * sizeof(PtcNode) ).c_str(), tree->GetPtcNodeCount() );
	//printf( "sizeof PtcCell Mem Pool = %s (%d elements) \n", getReadableSize( tree->GetPtcCellCount() * sizeof(PtcCell) ).c_str(), tree->GetPtcCellCount() );
	//printf( "sizeof asset Mem Pool = %s (%d elements) \n", getReadableSize( sizeof(Vec3) * 3 * tree->GetElementCount() * 2 ).c_str(), tree->GetElementCount() );

	// need to be in that order..
	// index assets
	rootNode->IndexAssets( tree->Instance()->GetElementBuffer() );

	// index data cells
	rootNode->IndexDataCells( tree->GetPtcCellBuffer() );

	// index the pointers
	rootNode->IndexPointers( rootNode );

	return tree;
}


void PtcTreeBuilder::FreeMemory()
{
	if ( m_PtcCells != NULL )
	{
		delete [] m_PtcCells;
		m_PtcCells = NULL;
	}
}

PtcCell * PtcTreeBuilder::GetPtcCell( int x, int y, int z )
{
	return &m_PtcCells[ x * m_SplitCount * m_SplitCount + y * m_SplitCount + z ];
}

bool PtcTreeBuilder::IsPtcCellRangeEmpty( int startX, int startY, int startZ, int endX, int endY, int endZ )
{
	for (int iX = startX; iX < endX; ++iX)
	{
		for (int iY = startY; iY < endY; ++iY)
		{
			for (int iZ = startZ; iZ < endZ; ++iZ)
			{
				if ( GetPtcCell(iX,iY,iZ)->GetCount() > 0 ) return false;
			}
		}
	}

	return true;
}

void PtcTreeBuilder::GetPtcCellsIndexers( const Imath::V3f & position, int & xPos, int & yPos, int & zPos )
{
	assert( VecBetween( position, m_Min, m_Max ) );

	xPos = (int)(( position.x - m_Min.x ) / m_SplitRange.x);
	yPos = (int)(( position.y - m_Min.y ) / m_SplitRange.y);
	zPos = (int)(( position.z - m_Min.z ) / m_SplitRange.z);

	assert(xPos <= m_SplitCount);
	if (xPos == m_SplitCount) xPos--; // on the max edge..
	assert(yPos <= m_SplitCount);
	if (yPos == m_SplitCount) yPos--; // on the max edge..
	assert(zPos <= m_SplitCount);
	if (zPos == m_SplitCount) zPos--; // on the max edge..
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
