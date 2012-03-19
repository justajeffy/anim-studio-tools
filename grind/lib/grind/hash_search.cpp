/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: $"
 * SVN_META_ID = "$Id: hash_search.cpp 45312 2010-09-09 06:11:02Z chris.cooper $"
 */

//-------------------------------------------------------------------------------------------------

#include <drdDebug/log.h>
DRD_MKLOGGER( L, "drd.grind.HashSearch" );

#include "hash_search.h"
#include "host_vector.h"


#include <algorithm>

using namespace grind;
using namespace drd;

extern "C"
void hashSearchBuildDevice( Imath::V3f* iCellDim,
                            int i_HashTableSize,
                            int i_PSize,
                            const Imath::V3f* i_P,
                            int* o_HashVal,
                            int* o_PIndices,
                            int* o_HashIndices );

extern "C"
void hashSearchFindNearestNeighbourDevice( const Imath::V3f* i_CellDim,
                                           int i_HashTableSize,
                                           int i_PSize,
                                           const Imath::V3f* i_P,
                                           int* i_HashVal,
                                           int* i_PIndices,
                                           int* i_HashIndices,
                                           int i_QueryPCount,
                                           const Imath::V3f* i_QueryP,
                                           int* o_ClosestIndices );


//-------------------------------------------------------------------------------------------------
HashSearch::HashSearch( Imath::V3f i_CellDim, int i_TableSize )
: m_CellDimension( i_CellDim )
, m_TableSize( i_TableSize )
{
	DRD_LOG_DEBUG( L, "constructing a hash search with table size " << m_TableSize );
}

//-------------------------------------------------------------------------------------------------
void HashSearch::dump() const
{
	DRD_LOG_INFO( L, "input point count:\t" << m_PIndices.size() );
	DRD_LOG_INFO( L, "cell dimension:\t" << m_CellDimension );
	DRD_LOG_INFO( L, "table size:\t" << m_TableSize );
	DRD_LOG_INFO( L, "m_PIndices:\t" );
		m_PIndices.dump();
	DRD_LOG_INFO( L, "m_HashVal:\t" );
		m_HashVal.dump();
	DRD_LOG_INFO( L, "m_HashIndices:\t" );
		m_HashIndices.dump();
}

//-------------------------------------------------------------------------------------------------
void HashSearch::build( const DeviceVector< Imath::V3f >& i_P )
{
	// make sure data is correct size
	m_HashVal.resize( i_P.size() );
	m_PIndices.resize( i_P.size() );
	m_HashIndices.resize( m_TableSize );

	hashSearchBuildDevice( &m_CellDimension,
	                       m_TableSize,
	                       i_P.size(),
	                       i_P.getDevicePtr(),
	                       m_HashVal.getDevicePtr(),
	                       m_PIndices.getDevicePtr(),
	                       m_HashIndices.getDevicePtr() );
}

//-------------------------------------------------------------------------------------------------
void HashSearch::findClosest(	const DeviceVector< Imath::V3f >& i_P,
								const DeviceVector< Imath::V3f >& i_QueryPoints,
								DeviceVector< int >& o_ResultIndices )
{
	// make sure result is correct size
	o_ResultIndices.resize( i_QueryPoints.size() );
	if( i_QueryPoints.size() == 0 ) return;

	hashSearchFindNearestNeighbourDevice( &m_CellDimension,
	           	                       m_TableSize,
	           	                       i_P.size(),
	           	                       i_P.getDevicePtr(),
	           	                       m_HashVal.getDevicePtr(),
	           	                       m_PIndices.getDevicePtr(),
	           	                       m_HashIndices.getDevicePtr(),
	           	                       i_QueryPoints.size(),
	           	                       i_QueryPoints.getDevicePtr(),
	           	                       o_ResultIndices.getDevicePtr() );

}

//-------------------------------------------------------------------------------------------------
void HashSearch::findClosestGold(	const DeviceVector< Imath::V3f >& i_P,
									const DeviceVector< Imath::V3f >& i_QueryPoints,
									DeviceVector< int >& o_ResultIndices )
{
	HostVector< Imath::V3f > h_P;
	HostVector< Imath::V3f > h_QueryPoints;
	HostVector< int > h_ResultIndices;

	i_P.getValue( h_P );
	i_QueryPoints.getValue( h_QueryPoints );
	h_ResultIndices.resize( h_QueryPoints.size() );
	if ( h_QueryPoints.size() == 0 ) return;

	// for each query point
	for ( int i = 0 ; i < h_QueryPoints.size() ; ++i )
	{
		const Imath::V3f& query_pt = h_QueryPoints[ i ];
		float min_dist_squared = 1.0f;
		int result_id = -1;
		// for each input point
		for ( int j = 0 ; j < h_P.size() ; ++j )
		{
			const Imath::V3f& p = h_P[ j ];
			Imath::V3f delta = p - query_pt;
			delta /= m_CellDimension; // allow for non uniform cells (ie check against elipsoid)
			float dist_squared = delta.dot( delta );
			if ( dist_squared < min_dist_squared )
			{
				min_dist_squared = dist_squared;
				result_id = j;
			}
		}
		h_ResultIndices[ i ] = result_id;
	}
	o_ResultIndices.setValue( h_ResultIndices );
}

//-------------------------------------------------------------------------------------------------
Imath::V3f genRandPoint()
{
	Imath::V3f result;

	float mag = 10.0f;
	float offset = -5.0f;

	result.x = float( rand() ) / RAND_MAX * mag + offset;
	result.y = float( rand() ) / RAND_MAX * mag + offset;
	result.z = float( rand() ) / RAND_MAX * mag + offset;
	return result;
}

//-------------------------------------------------------------------------------------------------
bool grind::testHashSearch()
{
	DRD_LOG_INFO( L, "testing HashSearch..." );
	HostVector< Imath::V3f > h_p, h_query_p;

	const size_t num_points = 10000;
	const size_t hash_table_size = PRIME_10000;
	const size_t num_query_points = 1000;

	LOGVAR( num_points );
	LOGVAR( num_query_points );

	Imath::V3f cell_dim( 0.2f, 0.2f, 0.2f );

	h_p.resize( num_points );
	std::generate( h_p.begin(), h_p.end(), genRandPoint );

	DeviceVector< Imath::V3f > d_p;
	d_p.setValue( h_p );

	HashSearch searcher( cell_dim, hash_table_size );
	searcher.build( d_p );

	h_query_p.resize( num_query_points );
	std::generate( h_query_p.begin(), h_query_p.end(), genRandPoint );
	DeviceVector< Imath::V3f > d_query_p;
	d_query_p.setValue( h_query_p );

	DeviceVector< int > d_result, gold_result;

	DRD_LOG_INFO( L, "finding closest via hash table..." );

	searcher.findClosest( d_p, d_query_p, d_result );

	DRD_LOG_INFO( L, "finding closest via reference implementation O(N*M)..." );

	searcher.findClosestGold( d_p, d_query_p, gold_result );

	bool success = gold_result == d_result;

	if ( success )
	{
		DRD_LOG_INFO( L, "SUCCESS" );
	}
	else
	{
		DRD_LOG_INFO( L, "FAIL" );
	}

	return success;
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
