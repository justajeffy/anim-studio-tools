/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: $"
 * SVN_META_ID = "$Id: mesh_subdivide.cpp 80147 2011-05-05 00:12:31Z chris.cooper $"
 */


#include <drdDebug/log.h>
DRD_MKLOGGER(L,"drd.grind.MeshSubdivide");

#include "grind/log.h"
#define __class__ "MeshSubdivide"

#include "mesh_subdivide.h"
#include "host_vector.h"
#include "mesh.h"
#include "guide_set.h"
#include "timer.h"
#include "device_vector_algorithms.h"
#include <bee/io/MeshLoader.h>

#define BUFFER_OFFSET(i) ((char *)NULL + (i))

#define MAX_SUBD_ITERATIONS 4

//-------------------------------------------------------------------------------------------------
using namespace grind;
using namespace drd;
using namespace subd;

namespace grind {
namespace subd {
	void buildFaceListDevice( size_t i_FaceCount, const int* i_Connectivity, const int* i_CumulativePolyVertCounts, grind::subd::Face* o_FaceList, int* o_VertFaceValence );
	int buildEdgeListDevice( size_t i_FaceCount, const int* i_Connectivity, const int* i_CumulativePolyVertCounts, Edge* o_Edges, int* o_VertEdgeValence );
	void buildSubdFaceTopologyDevice( size_t i_EdgeCount, const Edge* i_Edges, const int* i_Connectivity, const int* i_CumulativePolyVertCounts, size_t i_FacePointsOffset, size_t i_EdgePointsOffset, size_t i_VertPointsOffset, int* o_FaceVertIds  );
	template< typename T > void genFacePointsDevice( size_t i_FaceCount, const Face* i_Faces, const T* i_SrcP, size_t i_FacePointsOffset, size_t i_VertPointsOffset, int i_Stride, int i_Offset, T* o_DstP );
	template< typename T > void tweakVertPointsDevice(size_t i_VertCount, const int* i_VertFaceValence, const int* i_VertEdgeValence, size_t i_VertPointsOffset, int i_Stride, int i_Offset, T* o_DstP );
	template< typename T > void genEdgePointsDevice( size_t i_EdgeCount, const Edge* i_Edges, const T* i_SrcP, const int* i_VertFaceValence, const int* i_VertEdgeValence, size_t i_EdgePointsOffset, size_t i_VertPointsOffset, int i_Stride, int i_Offset, T* o_DstP );
	template< typename T > void genVertPointsDevice( BorderSubdType i_BorderSubdType, size_t i_VertCount, const T* i_SrcP, const int* i_VertFaceValence, const int* i_VertEdgeValence, size_t i_VertPointsOffset, int i_Stride, int i_Offset, T* o_DstP );

	//! struct representing topology of data (might be P, UV etc)
	struct TopologyData
	{
		size_t srcVertCount;
		DeviceVector< int > cumulativePolyVertCounts;
		DeviceVector< int > vertFaceValence;
		DeviceVector< int > vertEdgeValence;
		DeviceVector< grind::subd::Face > faceList;
		DeviceVector< grind::subd::Edge > edgeList;
		size_t facePointsOffset, edgePointsOffset, vertPointsOffset;

		TopologyData() : srcVertCount(0), facePointsOffset(0), edgePointsOffset(0), vertPointsOffset(0) {}
	};

	//! superset of data required for an iteration
	struct IterationData
	{
		// topology of the un-subdivided mesh
		TopologyData topoP;
		TopologyData topoN;
		// the subivided mesh
		DeviceMesh mesh;
		// the subdivided guides
		GuideSet guides;
	};
}
}

//-------------------------------------------------------------------------------------------------
MeshSubdivide::MeshSubdivide()
: m_Iterations(1)
{}

//-------------------------------------------------------------------------------------------------
MeshSubdivide::~MeshSubdivide()
{}

//-------------------------------------------------------------------------------------------------
void MeshSubdivide::setIterations( int i_Val )
{
	if( i_Val < 0 || i_Val > MAX_SUBD_ITERATIONS ){
		DRD_LOG_WARN( L, "currently only supporting 0-" << MAX_SUBD_ITERATIONS << " iterations" );
	}
	if( i_Val < 0 ) i_Val = 0;
	if( i_Val > MAX_SUBD_ITERATIONS ) i_Val = MAX_SUBD_ITERATIONS;
	m_Iterations = i_Val;
}

//-------------------------------------------------------------------------------------------------
void meshSanityCheck( const DeviceVector< int >& i_PolyVertCounts )
{
	int minFaceVerts = getMinValue( i_PolyVertCounts );
	int maxFaceVerts = getMaxValue( i_PolyVertCounts );
	if( minFaceVerts < 3 || maxFaceVerts > 4 ){
		throw std::runtime_error( "Only 3 and 4 sided faces allowed at the moment for MeshSubdivide" );
	}
}

//-------------------------------------------------------------------------------------------------
void updateTopology(	size_t i_SrcVertCount,
                    	const DeviceVector<int>& i_MeshConnectivity,
						const DeviceVector<int>& i_PolyVertCounts,
						subd::TopologyData& o_Topo,
						DeviceVector<int>* o_Id )
{
	o_Topo.srcVertCount = i_SrcVertCount;

	meshSanityCheck( i_PolyVertCounts );

	o_Topo.cumulativePolyVertCounts.resize( i_PolyVertCounts.size() );
	inclusiveScan( i_PolyVertCounts, o_Topo.cumulativePolyVertCounts );

	o_Topo.vertEdgeValence.resize( o_Topo.srcVertCount );
	setAllElements( 0, o_Topo.vertEdgeValence );
	o_Topo.vertFaceValence.resize( o_Topo.srcVertCount );
	setAllElements( 0, o_Topo.vertFaceValence );

	int face_count = i_PolyVertCounts.size();

	o_Topo.faceList.resize( face_count );

	buildFaceListDevice( face_count, i_MeshConnectivity.getDevicePtr(), o_Topo.cumulativePolyVertCounts.getDevicePtr(), o_Topo.faceList.getDevicePtr(), o_Topo.vertFaceValence.getDevicePtr() );

	int non_unique_edge_count = reduce( i_PolyVertCounts );

	DeviceVector< grind::subd::Edge > d_edges;
	d_edges.resize( non_unique_edge_count );
	int actual_edge_count = buildEdgeListDevice( face_count, i_MeshConnectivity.getDevicePtr(), o_Topo.cumulativePolyVertCounts.getDevicePtr(), d_edges.getDevicePtr(), o_Topo.vertEdgeValence.getDevicePtr() );
	o_Topo.edgeList.setValueDevice( d_edges.getDevicePtr(), actual_edge_count );

	int subd_quad_count = non_unique_edge_count;
	DRD_LOG_INFO( L, "faces before subd: " << face_count );
	DRD_LOG_INFO( L, "faces after subd: " << subd_quad_count );

	o_Topo.facePointsOffset = 0;
	o_Topo.edgePointsOffset = o_Topo.faceList.size();
	o_Topo.vertPointsOffset = o_Topo.faceList.size() + o_Topo.edgeList.size();

	if( o_Id != NULL ){
		o_Id->resize( subd_quad_count * 4, -1 );
		buildSubdFaceTopologyDevice( o_Topo.edgeList.size(), o_Topo.edgeList.getDevicePtr(), i_MeshConnectivity.getDevicePtr(), o_Topo.cumulativePolyVertCounts.getDevicePtr(), o_Topo.facePointsOffset, o_Topo.edgePointsOffset, o_Topo.vertPointsOffset, o_Id->getDevicePtr() );
	}
}

//-------------------------------------------------------------------------------------------------
template< typename T >
void clearValues( const DeviceVector< T >& i_Src, const subd::TopologyData& i_Topo, int i_Stride, int i_Offset, DeviceVector< T >& o_Dst  )
{
	size_t vert_count = i_Topo.faceList.size() + i_Topo.edgeList.size() + i_Topo.srcVertCount;

	if( i_Src.size() != i_Topo.srcVertCount * i_Stride ){
		throw std::runtime_error( "mesh topology doesn't appear to match guide topology" );
	}

	//! should provide a zero value for vec3, vec2, float etc
	const static T zero = T() * 0.0f;

	o_Dst.resize( vert_count * i_Stride );
	setAllElements( zero, o_Dst );
}

//-------------------------------------------------------------------------------------------------
template< typename T >
void updateValues( BorderSubdType i_BorderSubdType, const DeviceVector< T >& i_Src, const subd::TopologyData& i_Topo, int i_Stride, int i_Offset, DeviceVector< T >& o_Dst  )
{
	assert( o_Dst.size() != 0 );

	grind::subd::genFacePointsDevice( i_Topo.faceList.size(), i_Topo.faceList.getDevicePtr(), i_Src.getDevicePtr(), i_Topo.facePointsOffset, i_Topo.vertPointsOffset, i_Stride, i_Offset, o_Dst.getDevicePtr() );
	grind::subd::tweakVertPointsDevice( i_Topo.srcVertCount, i_Topo.vertFaceValence.getDevicePtr(), i_Topo.vertEdgeValence.getDevicePtr(), i_Topo.vertPointsOffset, i_Stride, i_Offset, o_Dst.getDevicePtr() );
	grind::subd::genEdgePointsDevice( i_Topo.edgeList.size(), i_Topo.edgeList.getDevicePtr(), i_Src.getDevicePtr(), i_Topo.vertFaceValence.getDevicePtr(), i_Topo.vertEdgeValence.getDevicePtr(), i_Topo.edgePointsOffset, i_Topo.vertPointsOffset, i_Stride, i_Offset, o_Dst.getDevicePtr() );
	grind::subd::genVertPointsDevice( i_BorderSubdType, i_Topo.srcVertCount, i_Src.getDevicePtr(), i_Topo.vertFaceValence.getDevicePtr(), i_Topo.vertEdgeValence.getDevicePtr(), i_Topo.vertPointsOffset, i_Stride, i_Offset, o_Dst.getDevicePtr() );
}

//-------------------------------------------------------------------------------------------------
void MeshSubdivide::buildIterData()
{
	if( m_IterData.size() != MAX_SUBD_ITERATIONS ){
		m_IterData.resize( MAX_SUBD_ITERATIONS );
		for( int i = 0; i < MAX_SUBD_ITERATIONS; ++i ){
			m_IterData[i].reset( new IterationData() );
		}
	}
}

//-------------------------------------------------------------------------------------------------
void MeshSubdivide::process( const DeviceMesh& i_Src, DeviceMesh& o_Dst )
{
	LOGFN0();

	// early exit
	if( m_Iterations == 0 ){
		o_Dst = i_Src;
		return;
	}

	bool iteration_changed = m_Iterations != m_PrevIterations;
	m_PrevIterations = m_Iterations;

	buildIterData();

	// DRD_LOG_VERBATIM( L, "iterations: " << m_Iterations );
	// CPU_SCOPE_TIMER( "process" );

	for( int i = 0; i < m_Iterations; ++i ){
		const DeviceMesh& src_mesh = i == 0 ? i_Src : m_IterData[i-1]->mesh;
		DeviceMesh& dst_mesh = i == (m_Iterations-1) ? o_Dst : m_IterData[i]->mesh;
		TopologyData& topo_p = m_IterData[i]->topoP;
		TopologyData& topo_n = m_IterData[i]->topoN;

		const DeviceVector<Imath::V3f>& src_P = src_mesh.getP();
		const DeviceVector<Imath::V3f>& src_N = src_mesh.getN();
		const DeviceVector<Imath::V2f>& src_UV = src_mesh.getUV();
		const DeviceVector<int>& src_ConP = src_mesh.getConnectivityP();
		const DeviceVector<int>& src_ConN = src_mesh.getConnectivityN();
		const DeviceVector<int>& src_ConUV = src_mesh.getConnectivityUV();
		const DeviceVector<int>& src_vert_counts = src_mesh.getPolyVertCounts();

		// create a dummy frame entry
		DeviceVector<Imath::V3f>& dst_P = dst_mesh.getP();
		DeviceVector<Imath::V3f>& dst_N = dst_mesh.getN();
		DeviceVector<Imath::V2f>& dst_UV = dst_mesh.getUV();
		DeviceVector<int>& dst_ConP = dst_mesh.getConnectivityP();
		DeviceVector<int>& dst_ConN = dst_mesh.getConnectivityN();
		DeviceVector<int>& dst_ConUV = dst_mesh.getConnectivityUV();
		DeviceVector<int>& dst_vert_counts = dst_mesh.getPolyVertCounts();

		bool needs_topo_update = topo_p.srcVertCount != src_P.size() || iteration_changed;
		if( needs_topo_update ){
			DRD_LOG_INFO( L, "updating mesh topology for iteration " << i );
			TopologyData topo_uv;

			updateTopology( src_P.size(), src_ConP, src_vert_counts, topo_p, &dst_ConP );
			updateTopology( src_N.size(), src_ConN, src_vert_counts, topo_n, &dst_ConN );
			updateTopology( src_UV.size(), src_ConUV, src_vert_counts, topo_uv, &dst_ConUV );

			// update vert counts
			dst_vert_counts.resize( dst_ConP.size() / 4 );
			setAllElements( 4, dst_vert_counts );

			// update uvs
			clearValues( src_UV, topo_uv, 1, 0, dst_UV );
			updateValues( BORDER_SUBD_NONE, src_UV, topo_uv, 1, 0, dst_UV );
		}

		// do these every frame
		{
			// update p
			clearValues( src_P, topo_p, 1, 0, dst_P );
			updateValues( BORDER_SUBD_UP_TO_EDGE, src_P, topo_p, 1, 0, dst_P );

			// update n
			clearValues( src_N, topo_n, 1, 0, dst_N );
			updateValues( BORDER_SUBD_UP_TO_EDGE, src_N, topo_n, 1, 0, dst_N );
		}

		dst_mesh.setTopologyChanged();
		dst_mesh.setSubdIterations( m_Iterations );
	}
}

//-------------------------------------------------------------------------------------------------
void MeshSubdivide::processGuides( const DeviceMesh& i_SrcMesh, const GuideSet& i_SrcGuides, GuideSet& o_DstGuides )
{
	//LOGFN0();

	// early exit
	if( m_Iterations == 0 ){
		o_DstGuides = i_SrcGuides;
		return;
	}

	bool iteration_changed = m_Iterations != m_PrevIterations;
	m_PrevIterations = m_Iterations;

	buildIterData();

	// DRD_LOG_VERBATIM( L, "iterations: " << m_Iterations );
	// CPU_SCOPE_TIMER( "processGuides" );

	for( int i = 0; i < m_Iterations; ++i ){
		const DeviceMesh& src_mesh = i == 0 ? i_SrcMesh : m_IterData[i-1]->mesh;
		DeviceMesh& dst_mesh = m_IterData[i]->mesh;
		const GuideSet& src_guides = i == 0 ? i_SrcGuides : m_IterData[i-1]->guides;
		GuideSet& dst_guides = i == (m_Iterations-1) ? o_DstGuides : m_IterData[i]->guides;
		TopologyData& topo_p = m_IterData[i]->topoP;
		TopologyData& topo_n = m_IterData[i]->topoN;

		const DeviceVector<Imath::V3f>& src_P_mesh = src_mesh.getP();
		const DeviceVector<int>& src_ConP_mesh = src_mesh.getConnectivityP();
		const DeviceVector<Imath::V3f>& src_P_guides = src_guides.getP();
		const DeviceVector<int>& src_vert_counts_mesh = src_mesh.getPolyVertCounts();

		DeviceVector<Imath::V3f>& dst_P_mesh = dst_mesh.getP();
		DeviceVector<int>& dst_ConP_mesh = dst_mesh.getConnectivityP();
		DeviceVector<Imath::V3f>& dst_P_guides = dst_guides.getP();
		DeviceVector<int>& dst_vert_counts_mesh = dst_mesh.getPolyVertCounts();

		bool needs_topo_update = topo_p.srcVertCount != src_P_mesh.size();

		if( needs_topo_update || iteration_changed ){
			DRD_LOG_INFO( L, "updating guide topology for iteration " << i );

			// update mesh topology
			updateTopology( src_P_mesh.size(), src_ConP_mesh, src_vert_counts_mesh, topo_p, &dst_ConP_mesh );

			// update vert counts
			dst_vert_counts_mesh.resize( dst_ConP_mesh.size() / 4 );
			setAllElements( 4, dst_vert_counts_mesh );
		}
		unsigned int n_src_curves = src_guides.getNCurves();
		unsigned int n_cvs = src_guides.getNCVs();
		int stride = n_cvs;

		// update p on mesh
		clearValues( src_P_mesh, topo_p, 1, 0, dst_P_mesh );
		updateValues( BORDER_SUBD_UP_TO_EDGE, src_P_mesh, topo_p, 1, 0, dst_P_mesh );

		// only needs to be done once (not for each offset)
		clearValues( src_P_guides, topo_p, stride, 0, dst_P_guides );
		dst_guides.resize( dst_P_guides.size() / n_cvs, n_cvs );

		// for each cv up a guide, update p on the guides
		for( int offset = 0; offset < stride; ++offset )
		{
			updateValues( BORDER_SUBD_UP_TO_EDGE, src_P_guides, topo_p, stride, offset, dst_P_guides );
		}

		// update across vector
		clearValues( src_guides.getAcross(), topo_p, 1, 0, dst_guides.getAcross() );
		updateValues( BORDER_SUBD_UP_TO_EDGE, src_guides.getAcross(), topo_p, 1, 0, dst_guides.getAcross() );
	}
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
