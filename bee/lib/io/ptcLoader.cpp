/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: http://svn.drd.int/drd/apps/bee/trunk/lib/io/objLoader.cpp $"
 * SVN_META_ID = "$Id: ptcLoader.cpp 107811 2011-10-15 06:12:38Z stephane.bertout $"
 */

#include "io/ptcLoader.h"

#include <drdDebug/log.h>
DRD_MKLOGGER(L,"drd.bee.io.PtcLoader");

// 3delight
#include "pointcloud.h"
#include <boost/filesystem.hpp>
#include <iostream>

#include "gl/Mesh.h"

#include "ptcNode.h"
#include "ptcCell.h"
#include "ptcTree.h"
#include "ptcTreeBuilder.h"

using namespace bee;
using namespace std;

//-------------------------------------------------------------------------------------------------
PtcLoader::PtcLoader()
: m_PointCount(0)
, m_PointCloudHandle( NULL )
{}

//-------------------------------------------------------------------------------------------------
PtcLoader::~PtcLoader()
{
	close();
}

//-------------------------------------------------------------------------------------------------
void PtcLoader::open( const std::string& i_FilePath, int i_Density )
{
	DRD_LOG_INFO( L, "Opening file: "+i_FilePath );
	m_FilePath = i_FilePath;
	m_Density = i_Density;

	if ( !boost::filesystem::exists( i_FilePath ) ) throw std::runtime_error( std::string( "PtcLoader: file doesn't exist " ) + i_FilePath );

	//m_PointCloudHandle = PtcSafeOpenPointCloudFile( i_FilePath.c_str() );
	m_PointCloudHandle = PtcOpenPointCloudFile( i_FilePath.c_str(), NULL, NULL, NULL );

	if ( !m_PointCloudHandle ) throw std::runtime_error( "PtcLoader failed to load point cloud" );

	PtcGetPointCloudInfo( m_PointCloudHandle, "nvars", &m_UserVarCount );

	float world_to_camera[4][4] = {{ 1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1} };
	PtcGetPointCloudInfo( m_PointCloudHandle, "world2eye", &world_to_camera[0] );

	Matrix invViewMatrix = Matrix( world_to_camera );
	invViewMatrix = invViewMatrix.transpose();
	Matrix viewMatrix = invViewMatrix.inverse();

    float format[3] = { 0, 0, 0 };
    PtcGetPointCloudInfo( m_PointCloudHandle, "format", format );
    m_BakeWidth = (int)format[0];
    m_BakeHeight = (int)format[1];
    m_BakeAspect = format[2];

	getPositionTr( m_BakeCamPosition, viewMatrix );
	getForwardVectorTr( m_BakeCamLookAt, viewMatrix );
	getUpVectorTr( m_BakeCamUp, viewMatrix );

	float world_to_ndc[4][4] = {{ 1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1} };
	PtcGetPointCloudInfo( m_PointCloudHandle, "world2ndc", &world_to_ndc[0] );
	m_ViewProjMatrix = Matrix( world_to_ndc );
	m_ViewProjMatrix = m_ViewProjMatrix.transpose();
	m_InvViewProjMatrix = m_ViewProjMatrix.inverse();

	PtcGetPointCloudInfo( m_PointCloudHandle, "datasize", &m_UserDataSize );
	DRD_LOG_INFO( L, "\tvars: " << m_UserVarCount << ", data size: " << m_UserDataSize );

	int ret;

	// allocate memory for variables information
	const char** varnames = NULL;
	const char** vartypes = NULL;

	// read in the names and types of the variables
	ret = PtcGetPointCloudInfo( m_PointCloudHandle, "varnames", &varnames );
	if( !ret ) throw std::runtime_error( "error accessing varnames" );

	ret = PtcGetPointCloudInfo( m_PointCloudHandle, "vartypes", &vartypes );
	if( !ret ) throw std::runtime_error( "error accessing vartypes" );

	for ( int i = 0 ; i < m_UserVarCount ; ++i )
	{
		DRD_LOG_INFO( L, "\tvar: " << varnames[ i ] << ", type: " << vartypes[ i ] );

		m_PtcUserVars.push_back( make_pair( string(varnames[ i ]), string( vartypes[ i ] ) ) );
	}

	PtcGetPointCloudInfo( m_PointCloudHandle, "npoints", &m_PointCount );
	if ( m_Density < 100 ) m_PointCount = m_PointCount * m_Density / 100;

	DRD_LOG_INFO( L, "\t" << m_PointCount << " points (" << m_PointCount / 1000000 << " M)" );
}

//-------------------------------------------------------------------------------------------------
void PtcLoader::read()
{
	DRD_LOG_INFO( L, "reading point data..." );

	m_DataMap.clear();

	int pointStep = 100 / m_Density;

	std::vector<float>& P = m_PositionVec;

	std::vector< half >& NR = m_DataMap["NR"];
	std::vector< half >& other = m_DataMap["other"];

	P.resize( m_PointCount * 3 );
	NR.resize( m_PointCount * 4 );
	other.resize( m_PointCount * m_UserDataSize );

	std::vector<float>::iterator pi( P.begin() );
	std::vector< half >::iterator nri( NR.begin() ), oi( other.begin() );

	float n[3], p[3];
	float r;

	if( m_UserDataSize > 0 )
	{
		float * usd = ( m_UserDataSize > 0 ) ? ( new float[m_UserDataSize] ) : ( NULL );

		for( int i = 0; i < m_PointCount; ++i, pi+=3 )
		{
			if ( i%10000==0)
			{
				printf(" %d%%", (int) (100 * (float)i/ (float)m_PointCount));
				printf("\r");
				flush(cout);
			}

			int ret = PtcReadDataPoint( m_PointCloudHandle, &(*pi), n, &r, usd );

			*nri++ = half(n[0]);
			*nri++ = half(n[1]);
			*nri++ = half(n[2]);
			*nri++ = half(r);

			for (int j = 0; j < m_UserDataSize; ++j) *oi++ = half( usd[j] );

			m_BoundingBox.update( Vec3( pi[0], pi[1], pi[2] ) );

			// skip some points ?
			for( int j = 1; j < pointStep; ++j ) PtcReadDataPoint( m_PointCloudHandle, p, n, &r, usd );
		}

		if ( usd != NULL ) delete [] usd;
	}
	else
	{
		for( int i = 0; i < m_PointCount; ++i, pi+=3 )
		{
			if ( i%10000==0)
			{
				printf(" %d%%", (int) (100 * (float)i/ (float)m_PointCount));
				printf("\r");
				flush(cout);
			}

			int ret = PtcReadDataPoint( m_PointCloudHandle, &(*pi), n, &r, NULL );

			*nri++ = half(n[0]);
			*nri++ = half(n[1]);
			*nri++ = half(n[2]);
			*nri++ = half(r);

			m_BoundingBox.update( Vec3( pi[0], pi[1], pi[2] ) );

			// skip some points ?
			for( int j = 1; j < pointStep; ++j ) PtcReadDataPoint( m_PointCloudHandle, p, n, &r, NULL );
		}
	}



	DRD_LOG_INFO( L, "read point data" );
}


//-------------------------------------------------------------------------------------------------
void PtcLoader::close()
{
	if( m_PointCloudHandle == NULL ) return;
	PtcClosePointCloudFile( m_PointCloudHandle );
}

//-------------------------------------------------------------------------------------------------
boost::shared_ptr< Mesh > PtcLoader::createMesh()
{
	boost::shared_ptr< Mesh > mesh( new Mesh( m_PointCount ) );

	std::vector<float>& P = m_PositionVec;
	std::vector<half>& N = m_DataMap["N"];
	//std::vector<float>& radius = m_DataMap["radius"];
	//std::vector<float>& other = m_DataMap["other"];

	mesh->createVertexBuffer( 3, sizeof( float ), &(m_DataMap["P"][0]) );
	mesh->createNormalBuffer( 3, sizeof( half ), &(m_DataMap["N"][0]) );

	return mesh;
}

// debugging !
template <typename T>
inline std::string ToString(T tX)
{
	std::ostringstream oStream;
	oStream << tX;
	return oStream.str();
}

std::string getReadableSize( int byte_count )
{
	if ( byte_count < 1024 ) return ToString( byte_count ) + std::string("b");
	else if ( byte_count < 1024 * 1024 ) return ToString( byte_count / 1024 ) + std::string("Kb");
	else return ToString( byte_count / (1024*1024) ) + std::string("Mb");
}

int add_done = 0;

Imath::V3f UnrecursiveCheckPositions( const PtcNode * rootNode )
{
	const Imath::V3f & Min = PtcTreeBuilder::Instance()->GetMin();
	PtcTree * tree = PtcTree::Instance();

	Imath::V3f diff( 0, 0, 0 );

	const int maxNodeStackCount = 64;
	const PtcNode * nodeStack[ maxNodeStackCount ];
	int nodeStackEnd = 0, nodeStackBegin = 0;
	const PtcNode * curNode = rootNode;

	nodeStack[ nodeStackEnd ++] = curNode;

    do
    {
    	//printf( "nodeStackBegin = %d | nodeStackEnd = %d \n", nodeStackBegin, nodeStackEnd );

    	//printf( " " );
    	//for (int i=nodeStackBegin; i<nodeStackEnd; ++i) printf("%d ", nodeStack[ i ] - rootNode );
    	//printf( " \n" );

    	/*{
			int i = nodeStackBegin;
			printf( " " );
			do
			{
				printf("%d[%d] ", nodeStack[ i ] - rootNode, i );
				i++;
				if (i == maxNodeStackCount) i = 0;
			}
			while ( i != (nodeStackEnd) );
			printf( " \n\n" );
    	}*/

    	curNode = nodeStack[ nodeStackBegin ++ ];
    	//printf( "curNode is %d \n", curNode - rootNode );
    	nodeStackBegin = nodeStackBegin % maxNodeStackCount;

    	if ( curNode->IsLeaf() )
    	{
    		//printf( "                                          process leaf %d !! \t\t dataCellIdx = %d \n", curNode - rootNode, curNode->GetDataCellIdx() );

    		const PtcCell * tdataCell = tree->GetPtcCell( curNode->GetDataCellIdx() );
    		const char * elements = ( tdataCell->GetElementsIdx() + tree->GetElementBuffer() );

    		for( int i = 0; i < tdataCell->GetCount(); ++i )
    		{
    			const Imath::V3f * position = ( const Imath::V3f * ) ( elements + SIZEOF_ELEMENT * i + POSITION_OFFSET );
    			diff += ( *position - Min );
    			add_done++;
    		}
    	}
    	else
    	{
    		const PtcNode * savedCurNode = curNode;

			// insert at the beginning to minimize the stack size !
    		// first count how many we'll have to add
    		int count2Add = 0;
    		if ( curNode->m_ChildrenIdx != -1 )
    		{
    			count2Add++;
    			curNode = &tree->m_PtcNodeBuffer[ curNode->m_ChildrenIdx ];
    	        while( curNode->GetNextIdx() != -1 )
    	        {
    	        	curNode = &tree->m_PtcNodeBuffer[ curNode->GetNextIdx() ];
    	        	count2Add++;
    	        }
    		}
    		curNode = savedCurNode;

			int insert_at = nodeStackEnd;
			bool copy_to_nodeStackEnd = true;
    		if ( count2Add > 0 && curNode != rootNode && nodeStackBegin != (nodeStackEnd - 1))
    		{
    			//printf("need to add %d \n", count2Add);

				int i = nodeStackEnd - 1;
				do
				{
					int newI = (i + count2Add) % maxNodeStackCount;
					nodeStack[newI] = nodeStack[i];
					nodeStack[i] = rootNode;
					i--;
					if (i < 0) i = maxNodeStackCount - 1;
				}
				while ( i != (nodeStackBegin) );
				int newI = (i + count2Add) % maxNodeStackCount;
				nodeStack[newI] = nodeStack[i];
				nodeStack[i] = rootNode;

				nodeStackEnd += count2Add;
				nodeStackEnd = nodeStackEnd % maxNodeStackCount;

				insert_at = nodeStackBegin;
				copy_to_nodeStackEnd = false;
    		}

			// add child
			curNode = &tree->m_PtcNodeBuffer[ curNode->m_ChildrenIdx ];
			nodeStack[ insert_at ++ ] = curNode;

			//printf( "add curNode children %d inserted @ %d!! \n", curNode - rootNode, insert_at );

			if ( insert_at ==  maxNodeStackCount )
			{
				//printf( "RESTART RING !!! nodeStackBegin = %d | nodeStackEnd = %d \n", nodeStackBegin, insert_at );
				insert_at = 0; // ring buffer
			}

			// add all the next nodes of this child
			while( curNode->GetNextIdx() != -1 )
			{
				curNode = &tree->m_PtcNodeBuffer[ curNode->GetNextIdx() ];
				nodeStack[ insert_at ++ ] = curNode;

				//printf( "add curNode next %d inserted @ %d!! \n", curNode - rootNode, insert_at );

				if ( insert_at ==  maxNodeStackCount )
				{
					//printf( "RESTART RING !!! nodeStackBegin = %d | nodeStackEnd = %d \n", nodeStackBegin, insert_at );
					insert_at = 0; // ring buffer
				}
			}

    	    if ( copy_to_nodeStackEnd == true) nodeStackEnd = insert_at;


    	}
    }
    while ( (nodeStackBegin) != nodeStackEnd );

	return diff;
}

// test / debug code
/*const PtcNode * PtcLoader::generatePtcTree( int split_count )
{
	PtcNode * rootNode = NULL;

	PtcTreeBuilder * treeBuilder = new PtcTreeBuilder( split_count, getBoundingBox().getMin(), getBoundingBox().getMax() );

	// extra kdtree test stuff
	std::vector< float > vertexVec;
	std::vector< float > normalVec;
	std::vector< float > radiusVec;
	vertexVec.reserve( getPointCount() * 3 );
	normalVec.reserve( getPointCount() * 3 );
	radiusVec.reserve( getPointCount() );
	fillVector( vertexVec, "P" );
	fillVector( normalVec, "N" );
	fillVector( radiusVec, "radius" );

	printf( "min = %f %f %f \n", treeBuilder->GetMin().x, treeBuilder->GetMin().y, treeBuilder->GetMin().z );
	printf( "max = %f %f %f \n", treeBuilder->GetMax().x, treeBuilder->GetMax().y, treeBuilder->GetMax().z );
	printf( "splitRange = %f %f %f \n", treeBuilder->GetSplitRange().x, treeBuilder->GetSplitRange().y, treeBuilder->GetSplitRange().z );

	printf("getPointCount() = %d \n", getPointCount());

	for ( int i = 0; i < getPointCount(); ++i )
	{
		Vec3 position( vertexVec[i*3], vertexVec[i*3+1], vertexVec[i*3+2] );
		Vec3 normal( normalVec[i*3], normalVec[i*3+1], normalVec[i*3+2] );
		//printf("%f %f %f \n", position.x, position.y, position.z);

		int xPos, yPos, zPos;
		treeBuilder->GetPtcCellsIndexers( position, xPos, yPos, zPos );

		PtcCell * ptcs = treeBuilder->GetPtcCell(xPos,yPos,zPos);

		ptcs->SetBox( xPos, yPos, zPos, treeBuilder->GetSplitRange(), treeBuilder->GetMin() );
		ptcs->Add( position, normal, radiusVec[i] );
	}

	// we can now free the vectors..
	vertexVec.clear();
	normalVec.clear();

	Vec3 diff( 0, 0, 0 ); Vec3 diff1( 0, 0, 0 ); Vec3 diff2( 0, 0, 0 ); Vec3 diff3( 0, 0, 0 );
	Vec3 diff4( 0, 0, 0 ); Vec3 diff5( 0, 0, 0 ); Vec3 diff6( 0, 0, 0 ); Vec3 diff7( 0, 0, 0 );

	int total = 0, valid_count = 0;
	for(int iSplitX = 0; iSplitX < split_count; ++iSplitX)
	{
		for(int iSplitY = 0; iSplitY < split_count; ++iSplitY)
		{
			for(int iSplitZ = 0; iSplitZ < split_count; ++iSplitZ)
			{
				PtcCell * ptcCell = treeBuilder->GetPtcCell( iSplitX, iSplitY, iSplitZ );
				total += ptcCell->GetCount();

				if ( ptcCell->GetCount() > 0 )
				{
					valid_count ++;

					for(int i=0; i<ptcCell->GetCount(); ++i)
					{
						diff += ( ptcCell->GetPosition( i ) - treeBuilder->GetMin() );
						add_done++;
					}
				}
			}
		}
	}

	printf("diff  = %f %f %f (%d)\n", diff.x, diff.y, diff.z, add_done );

	printf("total = %d | valid = %d | avg point count per cell = %d /// %d (%d)\n", total, valid_count, getPointCount() / valid_count, split_count*split_count*split_count, split_count );
	assert( total == getPointCount() );

	printf("Starting populating RootNode... \n" );

	rootNode = new PtcNode( treeBuilder->GetMin(), treeBuilder->GetMax(), 0, 0, 0, split_count, split_count, split_count );
	rootNode->Populate();

	PtcNode::add_done = 0;
	diff1 = rootNode->CheckPositions();
	printf("diff1 = %f %f %f (%d)\n", diff1.x, diff1.y, diff1.z, PtcNode::add_done );

	printf("point count from rootNode = %d\n", rootNode->GetPointCount() );
	assert( rootNode->GetPointCount() == getPointCount() );

	// now let's repack the asset in one big memory buffer !
	PtcTree * ptcTree = new PtcTree( getPointCount(), SIZEOF_ELEMENT ); // position - normal - radius

	ptcTree->AllocatePtcCellMemPool( valid_count );
	rootNode->RepackAssets();
	assert( ptcTree->IsEmpty() == true );
	assert( ptcTree->IsPtcCellMemPoolEmpty() == true );

	// we can now delete the huge array of PtcCell
	treeBuilder->FreeMemory();

	PtcNode::add_done = 0;
	diff2 = rootNode->CheckPositions();
	printf("diff2 = %f %f %f (%d)\n", diff2.x, diff2.y, diff2.z, PtcNode::add_done );
	assert(diff1 == diff2);

	// now let's repack all the ptc nodes in one mempool
	printf("ptcNodeCreatedCount = %d \n", PtcNode::ptcNodeCreatedCount );
	PtcTree::Instance()->AllocatePtcNodeMemPool( PtcNode::ptcNodeCreatedCount );
	PtcNode * newRootNode = ptcTree->AllocatePtcNode();
	rootNode->RepackTo( newRootNode );
	delete rootNode;
	rootNode = newRootNode;

	printf("ptcNodeCreatedCount = %d \n", PtcNode::ptcNodeCreatedCount );

	PtcNode::add_done = 0;
	diff3 = rootNode->CheckPositions();
	printf("diff3 = %f %f %f (%d)\n", diff3.x, diff3.y, diff3.z, PtcNode::add_done );
	assert(diff1 == diff3);

	assert( ptcTree->IsPtcNodeMemPoolEmpty() == true );

	printf( "sizeof PtcNode Mem Pool = %s (%d elements) \n", getReadableSize( ptcTree->GetPtcNodeCount() * sizeof(PtcNode) ).c_str(), ptcTree->GetPtcNodeCount() );
	printf( "sizeof PtcCell Mem Pool = %s (%d elements) \n", getReadableSize( ptcTree->GetPtcCellCount() * sizeof(PtcCell) ).c_str(), ptcTree->GetPtcCellCount() );
	printf( "sizeof asset Mem Pool = %s (%d elements) \n", getReadableSize( sizeof(Vec3) * 3 * ptcTree->GetElementCount() * 2 ).c_str(), ptcTree->GetElementCount() );

	// we check that the prev hierarchy always ends up correctly to the rootNode
	Assert( ptcTree->GetRootPtcNode() == rootNode ); // rootNode being allocated first.. it's always the first one
	Assert( rootNode->GetPrev() == NULL );
	for( int i = 1; i < PtcNode::ptcNodeCreatedCount; ++i )
	{
		const PtcNode * currNode = ptcTree->GetPtcNode(i);
		const PtcNode * currPrevNode = currNode->GetPrev();

		while (currPrevNode != rootNode)
		{
			currPrevNode = currPrevNode->GetPrev();
		}
	}

	// we check that the cells are contiguous & in order !
	for ( int i = 0; i < (PtcTree::Instance()->GetPtcCellCount()-1); ++i )
	{
		const PtcCell * curr = PtcTree::Instance()->GetPtcCell( i );
		const PtcCell * next = PtcTree::Instance()->GetPtcCell( i + 1 );
		Assert( curr->GetElement(0) < next->GetElement(0) );
	}

	// need to be in that order..
	// index assets
	rootNode->IndexAssets( ptcTree->Instance()->GetElementBuffer() );

	// index data cells
	rootNode->IndexDataCells( ptcTree->GetPtcCellBuffer() );

	// index the pointers
	rootNode->IndexPointers( rootNode );

	PtcNode::add_done = 0;
	diff4 = rootNode->CheckPositionsFromIndex();
	printf("diff4 = %f %f %f (%d)\n", diff4.x, diff4.y, diff4.z, PtcNode::add_done );
	assert(diff1 == diff4);

	// hierarchy check again but using index instead of pointers
	Assert( rootNode->GetPrevIdx() == -1 );
	for( int i = 1; i < PtcNode::ptcNodeCreatedCount; ++i )
	{
		const PtcNode * currNode = ptcTree->GetPtcNode( i );
		const PtcNode * currPrevNode = ptcTree->GetPtcNode( currNode->GetPrevIdx() );

		while (currPrevNode != rootNode)
		{
			currPrevNode = ptcTree->GetPtcNode( currPrevNode->GetPrevIdx() );
		}
	}

	// now move the memory to check that all the pointers have been correctly replaced by indices
	ptcTree->MovePtcCellMemPool();
	ptcTree->MovePtcNodeMemPool();
	ptcTree->MoveAllocatedBuffer();

	rootNode = (PtcNode *) ptcTree->GetRootPtcNode(); // the root has moved !!!
	PtcNode::add_done = 0;
	diff5 = rootNode->CheckPositionsFromIndex();
	printf("diff5 = %f %f %f (%d)\n", diff5.x, diff5.y, diff5.z, PtcNode::add_done );
	assert(diff1 == diff5);

	ptcTree->SaveToFile( m_FilePath + ".ocd" );

	// clear all
	ptcTree->ClearPtcCells();
	ptcTree->ClearPtcNodes();
	delete ptcTree;

	// reload all
	ptcTree = new PtcTree( m_FilePath + ".ocd" );

	// check it's still fine !
	rootNode = (PtcNode *) ptcTree->GetRootPtcNode(); // the root has moved !!!
	PtcNode::add_done = 0;
	diff6 = rootNode->CheckPositionsFromIndex();
	printf("diff6 = %f %f %f (%d)\n", diff6.x, diff6.y, diff6.z, PtcNode::add_done );
	assert(diff1 == diff6);

	add_done = 0;
	diff7 = UnrecursiveCheckPositions( rootNode );
	printf("diff7 = %f %f %f (%d)\n", diff7.x, diff7.y, diff7.z, add_done );
	//assert(diff6 == diff7);

	// check cell-node connection consistency !
	for ( int i = 0; i < ptcTree->GetPtcCellCount(); ++ i)
	{
		const PtcCell * currentCell = ptcTree->GetPtcCell( i );
		const PtcNode * nodeContainingThisCell = ptcTree->GetPtcNode( currentCell->GetNodeIdx() );
		Assert( nodeContainingThisCell->IsLeaf() == true );
		long int dataCellIdx = nodeContainingThisCell->GetDataCellIdx();
		Assert( dataCellIdx == i );

		//printf("%d | count = %d \n", i, currentCell->GetCount() );
	}

	delete treeBuilder;
	treeBuilder = NULL;

	return rootNode;
}*/

// end debugging !


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
