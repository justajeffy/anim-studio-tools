/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: $"
 * SVN_META_ID = "$Id: $"
 */

//-------------------------------------------------------------------------------------------------

#include <drdDebug/log.h>
DRD_MKLOGGER(L,"drd.grind.MeshCuda");

#include <OpenEXR/ImathVec.h>

#include <grind/algorithm/counting_iterator.h>
#include <grind/algorithm/for_each.h>

#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include <cutil_math.h>

using namespace drd;

namespace grind {
namespace mesh {

//-------------------------------------------------------------------------------------------------
// calculate the number of triangles based on a polygon vert count
struct num_tris: public thrust::unary_function< int, int >
{
	__host__ __device__
	int operator()( const int& i_PolyVertCount )
	{
		return i_PolyVertCount - 2;
	}
};

//-------------------------------------------------------------------------------------------------
int calcCumulativeTriCountDevice( size_t i_PolyCount, const int* i_PolyVertCounts, int* o_PolyTriCounts )
{
#ifdef __DEVICE_EMULATION__
	const int* src = i_PolyVertCounts;
	int* dst = o_PolyTriCounts;
#else
	thrust::device_ptr<const int> src( i_PolyVertCounts );
	thrust::device_ptr<int> dst( o_PolyTriCounts );
#endif

	thrust::transform( src, src + i_PolyCount, dst, num_tris() );
	thrust::inclusive_scan( dst, dst + i_PolyCount, dst );

	// return array size (total triangle count times 3)
	return (*(dst+i_PolyCount-1)) * 3;
}

//-------------------------------------------------------------------------------------------------
struct TriConnectivityFunctor
{
	TriConnectivityFunctor( const int* _CumulativeVertCounts, const int* _Connectivity, const int* _CumulativeTriCounts, int* _TriConnectivity )
	: i_CumulativeVertCounts( _CumulativeVertCounts )
	, i_Connectivity( _Connectivity )
	, i_CumulativeTriCounts( _CumulativeTriCounts )
	, o_TriConnectivity( _TriConnectivity )
	{}

	__host__ __device__
	void operator()( int i )
	{
		int vert_start_id = i == 0 ? 0 : i_CumulativeVertCounts[ i-1 ];
		int vert_finish_id = i_CumulativeVertCounts[ i ];

		int tri_id = i == 0 ? 0 : i_CumulativeTriCounts[ i-1 ];

		int n_verts = vert_finish_id - vert_start_id;

		for( int v = 2; v < n_verts; ++v, ++tri_id )
		{
			int vertexIdx0 = i_Connectivity[ vert_start_id ];
			int vertexIdx1 = i_Connectivity[ vert_start_id + v-1 ];
			int vertexIdx2 = i_Connectivity[ vert_start_id + v ];

			o_TriConnectivity[ tri_id * 3     ] = vertexIdx0;
			o_TriConnectivity[ tri_id * 3 + 1 ] = vertexIdx1;
			o_TriConnectivity[ tri_id * 3 + 2 ] = vertexIdx2;
		}
	}

	const int* i_CumulativeVertCounts;
	const int* i_Connectivity;
	const int* i_CumulativeTriCounts;
	int* o_TriConnectivity;
};

//-------------------------------------------------------------------------------------------------
void buildTriConnectivityDevice( size_t i_PolyCount, const int* i_CumulativeVertCounts, const int* i_Connectivity, const int* i_CumulativeTriCounts, int* o_TriConnectivity )
{
	grind::CountingIterator first( 0 );
	grind::CountingIterator last( i_PolyCount );
	TriConnectivityFunctor f( i_CumulativeVertCounts, i_Connectivity, i_CumulativeTriCounts, o_TriConnectivity );

	GRIND_FOR_EACH( first, last, f );
}

//-------------------------------------------------------------------------------------------------
template< typename T >
struct BuildDuplicatesFunctor
{
	BuildDuplicatesFunctor( const int* _DuplicateMap, const T* _Src, T* _Dst )
	: i_DuplicateMap( _DuplicateMap )
	, i_Src( _Src )
	, o_Dst( _Dst )
	{}

	__host__ __device__
	void operator()( int i )
	{
		// perform the mapping
		o_Dst[ i ] = i_Src[ i_DuplicateMap[ i ] ];
	}

	const int* i_DuplicateMap;
	const T* i_Src;
	T* o_Dst;
};

//-------------------------------------------------------------------------------------------------
template< typename T >
void buildDuplicatesDevice( size_t i_DuplicateCount, const int* i_DuplicateMap, const T* i_Src, T* o_Dst )
{
	grind::CountingIterator first( 0 );
	grind::CountingIterator last( i_DuplicateCount );
	BuildDuplicatesFunctor<T> f( i_DuplicateMap, i_Src, o_Dst );

	GRIND_FOR_EACH( first, last, f );
}

template void buildDuplicatesDevice<Imath::V3f>( size_t i_DuplicateCount, const int* i_DuplicateMap, const Imath::V3f* i_Src, Imath::V3f* o_Dst );
template void buildDuplicatesDevice<Imath::V2f>( size_t i_DuplicateCount, const int* i_DuplicateMap, const Imath::V2f* i_Src, Imath::V2f* o_Dst );

//-------------------------------------------------------------------------------------------------
struct ComputeTangentsFunctor
{
	ComputeTangentsFunctor( const int* _TangentVertId, const float3* _P, const float3* _N, float3* _T )
	: i_TangentVertId( _TangentVertId )
	, i_P( _P )
	, i_N( _N )
	, o_T( _T )
	{}

	__host__ __device__
	void operator()( int vert_id )
	{
		int tangent_vert_id = i_TangentVertId[vert_id];
		float3 t = normalize( i_P[ tangent_vert_id ] - i_P[ vert_id ] );
		float3 n = normalize( i_N[ vert_id ] );
		float3 bn = cross( t, n );
		o_T[ vert_id ] = normalize( cross( bn, n ) );
	}

	const int* i_TangentVertId;
	const float3* i_P;
	const float3* i_N;
	float3* o_T;
};


//-------------------------------------------------------------------------------------------------
void computeTangentsDevice( size_t i_VertCount, const int* i_TangentVertId, const Imath::V3f* i_P, const Imath::V3f* i_N, Imath::V3f* o_Tangent )
{
	grind::CountingIterator first( 0 );
	grind::CountingIterator last( i_VertCount );
	ComputeTangentsFunctor f( i_TangentVertId, (const float3*)i_P, (const float3*)i_N, (float3*)o_Tangent );

	GRIND_FOR_EACH( first, last, f );
}

}
}
