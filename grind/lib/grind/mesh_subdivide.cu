/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: $"
 * SVN_META_ID = "$Id: $"
 */

#include <drdDebug/log.h>
DRD_MKLOGGER(L,"drd.grind.MeshSubdivideCuda");

#include <grind/algorithm/for_each.h>
#include <grind/algorithm/atomic.h>
#include <grind/algorithm/counting_iterator.h>
#include <grind/mesh_subdivide_types.h>
#include <grind/type_traits.h>

#include <thrust/host_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <cutil_inline.h>
#include <cutil_math.h>


using namespace grind::subd;
using namespace drd;

namespace grind {
namespace subd {

//-------------------------------------------------------------------------------------------------
struct BuildFaceListFunctor
{
	__host__ __device__
	BuildFaceListFunctor(	const int* _Connectivity,
							const int* _CumulativePolyVertCounts,
							Face* _FaceList )
	: i_Connectivity( _Connectivity )
	, i_CumulativePolyVertCounts( _CumulativePolyVertCounts )
	, o_FaceList( _FaceList )
	{}

	__host__ __device__
	void operator()( int i )
	{
		int start_id = i == 0 ? 0 : i_CumulativePolyVertCounts[ i-1 ];
		int finish_id = i_CumulativePolyVertCounts[ i ];
		int n_verts = finish_id - start_id;

		grind::subd::Face& f = o_FaceList[i];

		f.v0 = i_Connectivity[start_id];
		f.v1 = i_Connectivity[start_id+1];
		f.v2 = i_Connectivity[start_id+2];
		if( n_verts > 3 ){
			f.v3 = i_Connectivity[start_id+3];
		}
		else
			f.v3 = -1;
	}

	const int* i_Connectivity;
	const int* i_CumulativePolyVertCounts;
	grind::subd::Face* o_FaceList;
};

//-------------------------------------------------------------------------------------------------
struct CalcVertFaceValenceFunctor
{
	__host__ __device__
	CalcVertFaceValenceFunctor( const Face* _Faces, int* _VertFaceValence )
	: i_Faces( _Faces )
	, o_VertFaceValence( _VertFaceValence )
	{}

	GRIND_HOST_DEVICE_ATOMIC
	void operator()( int i )
	{
		const Face& f = i_Faces[i];
		atomicAddT( 1, o_VertFaceValence[f.v0] );
		atomicAddT( 1, o_VertFaceValence[f.v1] );
		atomicAddT( 1, o_VertFaceValence[f.v2] );
		if( f.v3 != -1 ){
			atomicAddT( 1, o_VertFaceValence[f.v3] );
		}
	}

	const Face* i_Faces;
	int* o_VertFaceValence;
};

//-------------------------------------------------------------------------------------------------
void calcVertFaceValence( size_t i_FaceCount, const Face* i_Faces, int* o_VertFaceValence )
{
	grind::CountingIterator first( 0 );
	grind::CountingIterator last( i_FaceCount );

	CalcVertFaceValenceFunctor f( i_Faces, o_VertFaceValence );

	// note, not in parallel since we have atomic
	GRIND_FOR_EACH_ATOMIC( first, last, f );
}

//-------------------------------------------------------------------------------------------------
void buildFaceListDevice( size_t i_FaceCount, const int* i_Connectivity, const int* i_CumulativePolyVertCounts, grind::subd::Face* o_FaceList, int* o_VertFaceFalence )
{
	grind::CountingIterator first( 0 );
	grind::CountingIterator last( i_FaceCount );

	BuildFaceListFunctor f( i_Connectivity, i_CumulativePolyVertCounts, o_FaceList );

	GRIND_FOR_EACH( first, last, f );

	calcVertFaceValence( i_FaceCount, o_FaceList, o_VertFaceFalence );
}

//-------------------------------------------------------------------------------------------------
struct EdgeSortFunctor
{
	__host__ __device__
	bool operator()( const Edge& a, const Edge& b ) {
		// should sort by v0
		if( a.v0 < b.v0 ) return true;
		if( b.v0 < a.v0 ) return false;
		// then by v1
		return a.v1 < b.v1;
	}
};

//-------------------------------------------------------------------------------------------------
struct BuildEdgeListFunctor
{
	__host__ __device__
	BuildEdgeListFunctor( const int* _Connectivity, const int* _CumulativePolyVertCounts, Edge* _Edges )
	: i_Connectivity( _Connectivity )
	, i_CumulativePolyVertCounts( _CumulativePolyVertCounts )
	, o_Edges( _Edges )
	{}

	__host__ __device__
	void operator()( int i )
	{
		int start_id = i == 0 ? 0 : i_CumulativePolyVertCounts[i-1];
		int finish_id = i_CumulativePolyVertCounts[i];
		int n_verts = finish_id - start_id;

		int o_id = start_id;

		// for each vertex/edge in the face
		for( int v = 0; v < n_verts; ++v, ++o_id )
		{
			int va(-1),vb(-1);

			if( v == 0 ){
				va = i_Connectivity[ start_id + n_verts-1 ];
				vb = i_Connectivity[ start_id ];
			} else {
				va = i_Connectivity[ start_id + v - 1 ];
				vb = i_Connectivity[ start_id + v ];
			}

			Edge& e = o_Edges[o_id];
			// now order and store the result
			if( va < vb ){
				e.v0 = va;
				e.v1 = vb;
				e.f0 = i;
				e.f0_id = v;
			} else {
				e.v0 = vb;
				e.v1 = va;
				e.f0 = i;
				e.f0_id = v;
			}
		}
	}

	const int* i_Connectivity;
	const int* i_CumulativePolyVertCounts;
	Edge* o_Edges;
};

//-------------------------------------------------------------------------------------------------
//! share face information between edges based on adjacent edges in sorted list
struct ShareEdgeFacesFunctor
{
	__host__ __device__
	ShareEdgeFacesFunctor( int _EdgeCount, Edge* _Edges )
	: i_EdgeCount( _EdgeCount )
	, o_Edges( _Edges )
	{}

	__host__ __device__
	void process( Edge& a, const Edge& b )
	{
		// if verts match
		if( a.v0 == b.v0 && a.v1 == b.v1 ){
			// then set f1
			a.f1 = b.f0;
			a.f1_id = b.f0_id;
		}
	}

	__host__ __device__
	void operator()( int i )
	{
		Edge& a = o_Edges[i];

		if( i > 0 )
			process( a, o_Edges[i-1] );

		if( i < i_EdgeCount - 1 )
			process( a, o_Edges[i+1] );
	}

	int i_EdgeCount;
	Edge* o_Edges;
};

//-------------------------------------------------------------------------------------------------
struct CalcVertEdgeValenceFunctor
{
	__host__ __device__
	CalcVertEdgeValenceFunctor( const Edge* _Edges, int* _VertEdgeValence )
	: i_Edges( _Edges )
	, o_VertEdgeValence( _VertEdgeValence )
	{}

	GRIND_HOST_DEVICE_ATOMIC
	void operator()( int i )
	{
		const Edge& e = i_Edges[i];
		atomicAddT( 1, o_VertEdgeValence[e.v0] );
		atomicAddT( 1, o_VertEdgeValence[e.v1] );
	}

	const Edge* i_Edges;
	int* o_VertEdgeValence;
};

//-------------------------------------------------------------------------------------------------
void calcVertEdgeValence( size_t i_EdgeCount, const Edge* i_Edges, int* o_VertEdgeValence )
{
	grind::CountingIterator first( 0 );
	grind::CountingIterator last( i_EdgeCount );

	CalcVertEdgeValenceFunctor f( i_Edges, o_VertEdgeValence );

	// note, not in parallel since we have atomic
	GRIND_FOR_EACH_ATOMIC( first, last, f );
}

//-------------------------------------------------------------------------------------------------
int buildEdgeListDevice( size_t i_FaceCount, const int* i_Connectivity, const int* i_CumulativePolyVertCounts, Edge* o_Edges, int* o_VertEdgeValence )
{
#ifdef __DEVICE_EMULATION__
	const int* cpvc( i_CumulativePolyVertCounts );
	int total_edge_count = cpvc[ i_FaceCount-1 ];
	// currently having to make a duplicate of the list as thrust::sort didn't want to work with device_ptrs
	thrust::host_vector<Edge> edge_list( total_edge_count );
	Edge* edge_ptr( o_Edges );
#else
	thrust::device_ptr< const int > cpvc( i_CumulativePolyVertCounts );
	int total_edge_count = cpvc[ i_FaceCount-1 ];
	// currently having to make a duplicate of the list as thrust::sort didn't want to work with device_ptrs
	thrust::device_vector<Edge> edge_list( total_edge_count );
	thrust::device_ptr<Edge> edge_ptr( o_Edges );
#endif

	grind::CountingIterator first( 0 );
	grind::CountingIterator last( i_FaceCount );

	BuildEdgeListFunctor f( i_Connectivity, i_CumulativePolyVertCounts,
			thrust::raw_pointer_cast(&(*edge_list.begin())) );

	GRIND_FOR_EACH( first, last, f );

	EdgeSortFunctor es;
	thrust::sort( edge_list.begin(), edge_list.end(), es );

	ShareEdgeFacesFunctor sf( edge_list.size(), thrust::raw_pointer_cast(&(*edge_list.begin()) ) );
	GRIND_FOR_EACH( first, first + edge_list.size(), sf );

#ifdef __DEVICE_EMULATION__
	thrust::detail::normal_iterator< Edge* > new_end = thrust::unique( edge_list.begin(), edge_list.end() );
#else
	thrust::detail::normal_iterator< thrust::device_ptr<Edge> > new_end = thrust::unique( edge_list.begin(), edge_list.end() );
#endif
	size_t actual_edge_count = thrust::distance( edge_list.begin(), new_end );

	// copy result back
	thrust::copy( edge_list.begin(), edge_list.begin() + actual_edge_count, edge_ptr );

	calcVertEdgeValence( actual_edge_count, o_Edges, o_VertEdgeValence );

	return actual_edge_count;
}


//-------------------------------------------------------------------------------------------------
template< typename T >
struct GenFacePointsFunctor
{
	__host__ __device__
	GenFacePointsFunctor( const Face* _Faces, const T* _SrcP, size_t _FacePointsOffset, size_t _VertPointsOffset, int _Stride, int _Offset, T* _DstP )
	: i_Faces( _Faces )
	, i_SrcP( _SrcP )
	, i_FacePointsOffset( _FacePointsOffset )
	, i_VertPointsOffset( _VertPointsOffset )
	, i_Stride( _Stride )
	, i_Offset( _Offset )
	, o_DstP( _DstP )
	{}

	GRIND_HOST_DEVICE_ATOMIC
	void operator()( int i )
	{
		const Face& f = i_Faces[i];

		// sum first three verts
		T fp = i_SrcP[ f.v0 * i_Stride + i_Offset ] + i_SrcP[ f.v1 * i_Stride + i_Offset ] + i_SrcP[ f.v2 * i_Stride + i_Offset ];

		if( f.v3 == -1 ){
			// if vert count is 3
			fp *= 0.3333333f;
		} else {
			// otherwise
			fp += i_SrcP[ f.v3 * i_Stride + i_Offset ];
			fp *= 0.25f;
		}
		// set the face point
		o_DstP[ (i_FacePointsOffset + i) * i_Stride + i_Offset ] = fp;

		// add to the vert points
		atomicAddT( fp, o_DstP[ (i_VertPointsOffset + f.v0) * i_Stride + i_Offset ] );
		atomicAddT( fp, o_DstP[ (i_VertPointsOffset + f.v1) * i_Stride + i_Offset ] );
		atomicAddT( fp, o_DstP[ (i_VertPointsOffset + f.v2) * i_Stride + i_Offset ] );
		if( f.v3 != -1 ){
			atomicAddT( fp, o_DstP[ (i_VertPointsOffset + f.v3) * i_Stride + i_Offset ] );
		}
	}

	const Face* i_Faces;
	const T* i_SrcP;
	size_t i_FacePointsOffset;
	size_t i_VertPointsOffset;
	int i_Stride;
	int i_Offset;
	T* o_DstP;
};


//-------------------------------------------------------------------------------------------------
template< typename GRIND_TYPE >
void genFacePointsDevice( size_t i_FaceCount, const Face* i_Faces, const GRIND_TYPE* i_SrcP, size_t i_FacePointsOffset, size_t i_VertPointsOffset, int i_Stride, int i_Offset, GRIND_TYPE* o_DstP )
{
	grind::CountingIterator first( 0 );
	grind::CountingIterator last( i_FaceCount );

	typedef typename grind::type_traits< GRIND_TYPE >::cuda_type cuda_type;

	GenFacePointsFunctor<cuda_type> f( i_Faces, (cuda_type*)i_SrcP, i_FacePointsOffset, i_VertPointsOffset, i_Stride, i_Offset, (cuda_type*)o_DstP );

	GRIND_FOR_EACH_ATOMIC( first, last, f );
}

// explicit instantiations
template void genFacePointsDevice<Imath::V3f>( size_t i_FaceCount, const Face* i_Faces, const Imath::V3f* i_SrcP, size_t i_FacePointsOffset, size_t i_VertPointsOffset, int i_Stride, int i_Offset, Imath::V3f* o_DstP );
template void genFacePointsDevice<Imath::V2f>( size_t i_FaceCount, const Face* i_Faces, const Imath::V2f* i_SrcP, size_t i_FacePointsOffset, size_t i_VertPointsOffset, int i_Stride, int i_Offset, Imath::V2f* o_DstP );
template void genFacePointsDevice<float>( size_t i_FaceCount, const Face* i_Faces, const float* i_SrcP, size_t i_FacePointsOffset, size_t i_VertPointsOffset, int i_Stride, int i_Offset, float* o_DstP );

//! based on the valences, is the vert on a border?
__host__ __device__
inline bool isBorderVert( int i_FaceValence, int i_EdgeValence ){
	return i_FaceValence != i_EdgeValence;
}

//-------------------------------------------------------------------------------------------------
template< typename T >
struct TweakVertPointsFunctor
{
	__host__ __device__
	TweakVertPointsFunctor( const int* _VertFaceValence, const int* _VertEdgeValence, size_t _VertPointsOffset, int _Stride, int _Offset, T* _DstP )
	: i_VertFaceValence( _VertFaceValence )
	, i_VertEdgeValence( _VertEdgeValence )
	, i_VertPointsOffset( _VertPointsOffset )
	, i_Stride( _Stride )
	, i_Offset( _Offset )
	, o_DstP( _DstP )
	{}

	__host__ __device__
	void operator()( int i )
	{
		T& v = o_DstP[ (i_VertPointsOffset + i) * i_Stride + i_Offset ];

		if( isBorderVert( i_VertFaceValence[i], i_VertEdgeValence[i] ) ){
			// ignore face vert contribution for border verts
			v *= 0.0f;
		}
#if 0
		else {
			v *= float(i_VertEdgeValence[i]) / float(i_VertFaceValence[i]);
		}
#endif
	}

	const int* i_VertFaceValence;
	const int* i_VertEdgeValence;
	size_t i_VertPointsOffset;
	int i_Stride;
	int i_Offset;
	T* o_DstP;
};

//-------------------------------------------------------------------------------------------------
template< typename GRIND_TYPE >
void tweakVertPointsDevice( size_t i_VertCount, const int* i_VertFaceValence, const int* i_VertEdgeValence, size_t i_VertPointsOffset, int i_Stride, int i_Offset, GRIND_TYPE* o_DstP )
{
	grind::CountingIterator first( 0 );
	grind::CountingIterator last( i_VertCount );

	typedef typename grind::type_traits< GRIND_TYPE >::cuda_type cuda_type;

	TweakVertPointsFunctor<cuda_type> f( i_VertFaceValence, i_VertEdgeValence, i_VertPointsOffset, i_Stride, i_Offset, (cuda_type*)o_DstP );

	GRIND_FOR_EACH( first, last, f );
}

// explicit instantiations
template void tweakVertPointsDevice<Imath::V3f>( size_t i_VertCount, const int* i_VertFaceValence, const int* i_VertEdgeValence, size_t i_VertPointsOffset, int i_Stride, int i_Offset, Imath::V3f* o_DstP );
template void tweakVertPointsDevice<Imath::V2f>( size_t i_VertCount, const int* i_VertFaceValence, const int* i_VertEdgeValence, size_t i_VertPointsOffset, int i_Stride, int i_Offset, Imath::V2f* o_DstP );
template void tweakVertPointsDevice<float>( size_t i_VertCount, const int* i_VertFaceValence, const int* i_VertEdgeValence, size_t i_VertPointsOffset, int i_Stride, int i_Offset, float* o_DstP );

//-------------------------------------------------------------------------------------------------
template< typename T >
struct GenEdgePointsFunctor
{
	__host__ __device__
	GenEdgePointsFunctor( const Edge* _Edges, const T* _SrcP, const int* _VertFaceValence, const int* _VertEdgeValence, size_t _EdgePointsOffset, size_t _VertPointsOffset, int _Stride, int _Offset, T* _DstP )
	: i_Edges( _Edges )
	, i_SrcP( _SrcP )
	, i_VertFaceValence( _VertFaceValence )
	, i_VertEdgeValence( _VertEdgeValence )
	, i_EdgePointsOffset( _EdgePointsOffset )
	, i_VertPointsOffset( _VertPointsOffset )
	, i_Stride( _Stride )
	, i_Offset( _Offset )
	, o_DstP( _DstP )
	{}

	GRIND_HOST_DEVICE_ATOMIC
	void operator()( int i )
	{
		const Edge& e = i_Edges[i];

		// midpoint of edge
		T mp =  ( i_SrcP[ e.v0 * i_Stride + i_Offset ] + i_SrcP[ e.v1 * i_Stride + i_Offset ] ) * 0.5f; // * 0.5f * 2.0f;
		mp *= 2.0f;

		if( e.f1 == -1 ){
			// always add open edges to vert points
			atomicAddT( mp, o_DstP[ (i_VertPointsOffset + e.v0) * i_Stride + i_Offset ] );
			atomicAddT( mp, o_DstP[ (i_VertPointsOffset + e.v1) * i_Stride + i_Offset ] );
		} else {
			if( !isBorderVert( i_VertFaceValence[ e.v0 ], i_VertEdgeValence[ e.v0 ] ) ){
				atomicAddT( mp, o_DstP[ (i_VertPointsOffset + e.v0) * i_Stride + i_Offset ] );
			}
			if( !isBorderVert( i_VertFaceValence[ e.v1 ], i_VertEdgeValence[ e.v1 ] ) ){
				atomicAddT( mp, o_DstP[ (i_VertPointsOffset + e.v1) * i_Stride + i_Offset ] );
			}
		}

		// calculate the edge point
		T ep;

		if( e.f1 == -1 ){
#if 0
			ep = ( i_SrcP[ e.v0 ] + i_SrcP[ e.v1 ] + o_DstP[ e.f0 ] ) * 0.333333f;
#else
			// edge point ignores faces (is midpoint of end verts)
			ep = ( i_SrcP[ e.v0 * i_Stride + i_Offset ] + i_SrcP[ e.v1 * i_Stride + i_Offset ] ) * 0.5f;
#endif
		} else {
			// edge point takes into account face points
			ep = ( i_SrcP[ e.v0 * i_Stride + i_Offset ] + i_SrcP[ e.v1 * i_Stride + i_Offset ] + o_DstP[ e.f0 * i_Stride + i_Offset ] + o_DstP[ e.f1 * i_Stride + i_Offset ] ) * 0.25f;
		}
		o_DstP[ (i_EdgePointsOffset + i) * i_Stride + i_Offset ] = ep;

//		ep *= 2.0f;

	}

	const Edge* i_Edges;
	const T* i_SrcP;
	const int* i_VertFaceValence;
	const int* i_VertEdgeValence;
	size_t i_EdgePointsOffset;
	size_t i_VertPointsOffset;
	int i_Stride;
	int i_Offset;
	T* o_DstP;
};

//-------------------------------------------------------------------------------------------------
template< typename GRIND_TYPE >
void genEdgePointsDevice( size_t i_EdgeCount, const Edge* i_Edges, const GRIND_TYPE* i_SrcP, const int* i_VertFaceValence, const int* i_VertEdgeValence, size_t i_EdgePointsOffset, size_t i_VertPointsOffset, int i_Stride, int i_Offset, GRIND_TYPE* o_DstP )
{
	grind::CountingIterator first( 0 );
	grind::CountingIterator last( i_EdgeCount );

	typedef typename grind::type_traits< GRIND_TYPE >::cuda_type cuda_type;

	GenEdgePointsFunctor<cuda_type> f( i_Edges, (cuda_type*)i_SrcP, i_VertFaceValence, i_VertEdgeValence, i_EdgePointsOffset, i_VertPointsOffset, i_Stride, i_Offset, (cuda_type*)o_DstP );

	GRIND_FOR_EACH_ATOMIC( first, last, f );
}

// explicit instantiations
template void genEdgePointsDevice<Imath::V3f>( size_t i_EdgeCount, const Edge* i_Edges, const Imath::V3f* i_SrcP, const int* i_VertFaceValence, const int* i_VertEdgeValence, size_t i_EdgePointsOffset, size_t i_VertPointsOffset, int i_Stride, int i_Offset, Imath::V3f* o_DstP );
template void genEdgePointsDevice<Imath::V2f>( size_t i_EdgeCount, const Edge* i_Edges, const Imath::V2f* i_SrcP, const int* i_VertFaceValence, const int* i_VertEdgeValence, size_t i_EdgePointsOffset, size_t i_VertPointsOffset, int i_Stride, int i_Offset, Imath::V2f* o_DstP );
template void genEdgePointsDevice<float>( size_t i_EdgeCount, const Edge* i_Edges, const float* i_SrcP, const int* i_VertFaceValence, const int* i_VertEdgeValence, size_t i_EdgePointsOffset, size_t i_VertPointsOffset, int i_Stride, int i_Offset, float* o_DstP );

//-------------------------------------------------------------------------------------------------
template< typename T >
struct GenVertPointsFunctor
{
	__host__ __device__
	GenVertPointsFunctor( BorderSubdType _BorderSubdType, const T* _SrcP, const int* _VertFaceValence, const int* _VertEdgeValence, size_t _VertPointsOffset, int _Stride, int _Offset, T* _DstP )
	: i_BorderSubdType( _BorderSubdType )
	, i_SrcP( _SrcP )
	, i_VertFaceValence( _VertFaceValence )
	, i_VertEdgeValence( _VertEdgeValence )
	, i_VertPointsOffset( _VertPointsOffset )
	, i_Stride( _Stride )
	, i_Offset( _Offset )
	, o_DstP( _DstP )
	{}

	__host__ __device__
	void operator()( int i )
	{
		T vin = i_SrcP[ i * i_Stride + i_Offset ];
		int vc = i_VertEdgeValence[ i ];
		T vout;

		// if we're on an open boundary (as per prman interpolateboundary)
		if( isBorderVert( i_VertFaceValence[i], i_VertEdgeValence[i] ) ){
			switch( i_BorderSubdType ){
				case BORDER_SUBD_UP_TO_EDGE:
				{
					// incoming vert has been set to (sum of edge midpoints) x 2
					//o_DstP[ (i_VertPointsOffset + i) * i_Stride + i_Offset ] *= 1.0f / float(vc*2);
					vout = vin * 4.0f + o_DstP[ (i_VertPointsOffset + i) * i_Stride + i_Offset ];
					o_DstP[ (i_VertPointsOffset + i) * i_Stride + i_Offset ] = vout * 0.125f;
				}
					break;
				default:
				{
					o_DstP[ (i_VertPointsOffset + i) * i_Stride + i_Offset ] = vin;
				}
					break;
			}
			return;
		}

		// "vout initially contains the sum of the original vertex, all face points as well as twice the midpoints of all edges"
		vout = vin + o_DstP[ (i_VertPointsOffset + i) * i_Stride + i_Offset ];

		o_DstP[ (i_VertPointsOffset + i) * i_Stride + i_Offset ] = vin * (vc-3) / vc + (vout-vin) / (vc * vc);

//		o_DstP[ i_VertPointsOffset + i ] = ( vin * (vc-2) + (vout-vin) / vc ) /vc;
//		o_DstP[ i_VertPointsOffset + i ] = (vout + vin * float(vc-3)) / float(vc) * 0.33333;
//		o_DstP[ i_VertPointsOffset + i ] = ( vin * float(vc-3) / float(vc) + vout ) * 0.33333;
	}
	BorderSubdType i_BorderSubdType;
	const T* i_SrcP;
	const int* i_VertFaceValence;
	const int* i_VertEdgeValence;
	size_t i_VertPointsOffset;
	int i_Stride;
	int i_Offset;
	T* o_DstP;
};

//-------------------------------------------------------------------------------------------------
template< typename GRIND_TYPE >
void genVertPointsDevice( BorderSubdType i_BorderSubdType, size_t i_VertCount, const GRIND_TYPE* i_SrcP, const int* i_VertFaceValence, const int* i_VertEdgeValence, size_t i_VertPointsOffset, int i_Stride, int i_Offset, GRIND_TYPE* o_DstP )
{
	grind::CountingIterator first( 0 );
	grind::CountingIterator last( i_VertCount );

	typedef typename grind::type_traits< GRIND_TYPE >::cuda_type cuda_type;

	GenVertPointsFunctor<cuda_type> f( i_BorderSubdType, (cuda_type*)i_SrcP, i_VertFaceValence, i_VertEdgeValence, i_VertPointsOffset, i_Stride, i_Offset, (cuda_type*)o_DstP );

	GRIND_FOR_EACH( first, last, f );
}

// explicit instantiations
template void genVertPointsDevice<Imath::V3f>( BorderSubdType i_BorderSubdType, size_t i_VertCount, const Imath::V3f* i_SrcP, const int* i_VertFaceValence, const int* i_VertEdgeValence, size_t i_VertPointsOffset, int i_Stride, int i_Offset, Imath::V3f* o_DstP );
template void genVertPointsDevice<Imath::V2f>( BorderSubdType i_BorderSubdType, size_t i_VertCount, const Imath::V2f* i_SrcP, const int* i_VertFaceValence, const int* i_VertEdgeValence, size_t i_VertPointsOffset, int i_Stride, int i_Offset, Imath::V2f* o_DstP );
template void genVertPointsDevice<float>( BorderSubdType i_BorderSubdType, size_t i_VertCount, const float* i_SrcP, const int* i_VertFaceValence, const int* i_VertEdgeValence, size_t i_VertPointsOffset, int i_Stride, int i_Offset, float* o_DstP );

//-------------------------------------------------------------------------------------------------
struct BuildSubdFaceTopologyFunctor
{
	BuildSubdFaceTopologyFunctor( size_t _EdgeCount, const Edge* _Edges, const int* _Connectivity, const int* _CumulativePolyVertCounts, size_t _FacePointsOffset, size_t _EdgePointsOffset, size_t _VertPointsOffset, int* _FaceVertIds )
	: i_EdgeCount( _EdgeCount )
	, i_Edges( _Edges )
	, i_Connectivity( _Connectivity )
	, i_CumulativePolyVertCounts( _CumulativePolyVertCounts )
	, i_FacePointsOffset( _FacePointsOffset )
	, i_EdgePointsOffset( _EdgePointsOffset )
	, i_VertPointsOffset( _VertPointsOffset )
	, o_FaceVertIds( _FaceVertIds )
	{}

	__host__ __device__
	void buildFace( int i_FaceId, int i_LocalEdgeId, int i_EdgeId )
	{
		int start_id = i_FaceId == 0 ? 0 : i_CumulativePolyVertCounts[ i_FaceId-1 ];
		int finish_id = i_CumulativePolyVertCounts[i_FaceId];
		int n_verts = finish_id - start_id;
		int va(-1);

		int out_face_id = start_id + i_LocalEdgeId;
		int out_next_face_id = start_id + ((i_LocalEdgeId+1)%n_verts);

		if( i_LocalEdgeId == 0 ){
			va = i_Connectivity[ start_id + n_verts-1 ];
		} else {
			va = i_Connectivity[ start_id + i_LocalEdgeId - 1 ];
		}

		// update 3 verts of output quad using...
		// face point as vert0
		o_FaceVertIds[ out_face_id * 4 ] = i_FaceId + i_FacePointsOffset;
		// edge point as vert 1
		o_FaceVertIds[ out_face_id * 4 + 3 ] = i_EdgeId + i_EdgePointsOffset;
		// vert point as vert 2
		o_FaceVertIds[ out_face_id * 4 + 2 ] = va + i_VertPointsOffset;

		// update 1 vert of the next quad using...
		// edge point as vert 3
		o_FaceVertIds[ out_next_face_id * 4 + 1 ] = i_EdgeId + i_EdgePointsOffset;

	}

	__host__ __device__
	void operator()( int edge_id )
	{
		const Edge& e = i_Edges[ edge_id ];
		buildFace( e.f0, int(e.f0_id), edge_id );
		if( e.f1 != -1 ){
			buildFace( e.f1, int(e.f1_id), edge_id );
		}

	}

	size_t i_EdgeCount;
	const Edge* i_Edges;
	const int* i_Connectivity;
	const int* i_CumulativePolyVertCounts;
	size_t i_FacePointsOffset;
	size_t i_EdgePointsOffset;
	size_t i_VertPointsOffset;
	int* o_FaceVertIds;
};

//-------------------------------------------------------------------------------------------------
void buildSubdFaceTopologyDevice( size_t i_EdgeCount, const Edge* i_Edges, const int* i_Connectivity, const int* i_CumulativePolyVertCounts, size_t i_FacePointsOffset, size_t i_EdgePointsOffset, size_t i_VertPointsOffset, int* o_FaceVertIds  )
{
	grind::CountingIterator first( 0 );
	grind::CountingIterator last( i_EdgeCount );

	BuildSubdFaceTopologyFunctor f( i_EdgeCount, i_Edges, i_Connectivity, i_CumulativePolyVertCounts, i_FacePointsOffset, i_EdgePointsOffset, i_VertPointsOffset, o_FaceVertIds );

	GRIND_FOR_EACH( first, last, f );
}


}
}
