/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: $"
 * SVN_META_ID = "$Id: mesh.cpp 99443 2011-08-30 01:14:50Z hugh.rayner $"
 */

//-------------------------------------------------------------------------------------------------

#include <drdDebug/log.h>
#include <drdDebug/runtimeError.h>
DRD_MKLOGGER(L,"drd.grind.bindings.Mesh");

#include "mesh.h"
#include "device_vector_algorithms.h"

#include "utils.h"

// bee includes
#include <bee/io/Loader.h>
#include <bee/io/MeshLoader.h>
#include <bee/io/VacLoader.h>
#include <iostream>
#include <sstream>

#include <boost/foreach.hpp>

//-------------------------------------------------------------------------------------------------
using namespace grind;
using namespace drd;

namespace grind {
namespace mesh {
	int calcCumulativeTriCountDevice( size_t i_PolyCount, const int* i_PolyVertCounts, int* o_PolyTriCounts );
	void buildTriConnectivityDevice( size_t i_PolyCount, const int* i_CumulativeVertCounts, const int* i_Connectivity, const int* i_CumulativeVertCounts, int* o_TriConnectivity );
	template< typename T > void buildDuplicatesDevice( size_t i_DuplicateCount, const int* i_DuplicateMap, const T* i_Src, T* o_Dst );
	void computeTangentsDevice( size_t i_VertCount, const int* i_TangentVertId, const Imath::V3f* i_P, const Imath::V3f* i_N, Imath::V3f* o_Tangent );
}
}

using namespace mesh;

//-------------------------------------------------------------------------------------------------
HostMesh::HostMesh()
	:
	m_autoApplyTransforms( false )
,	m_referenceFrame( 0 )
{}


void
GrabData
(
	const std::vector< Imath::V3f > & a_P
,	const std::vector< bee::MeshPolygon > & a_Poly
,	const std::vector< Imath::V3f > & a_N
,	grind::HostVector< Imath::V3f > & o_P
,	grind::HostVector< Imath::V3f > & o_N
)
{
	o_P.assign( a_P.begin(), a_P.end() );

	// calculate vert normals by averaging face normals
	std::fill( o_N.begin(), o_N.end(), Imath::V3f(0,0,0) );

    BOOST_FOREACH( const bee::MeshPolygon & poly, a_Poly )
	{
		int nv = poly.m_FaceVector.size();
		for( int v = 0; v < nv; ++v )
		{
			int vert_id = poly.m_FaceVector[ v ].vertexIdx;
			int norm_id = poly.m_FaceVector[ v ].normalIdx;
			o_N[ vert_id ] += a_N[ norm_id ];
		}
    }
	for ( size_t i = 0 ; i < o_N.size() ; ++i )
	{
		o_N[ i ].normalize();
	}
}

//-------------------------------------------------------------------------------------------------
void HostMesh::read( const std::string & file_path )
{
	DRD_LOG_DEBUG( L, "read(" << file_path << ")" );
	m_Loader = bee::Loader::CreateMeshLoader( file_path );
	setAutoApplyTransforms( m_autoApplyTransforms );
	setReferenceFrame( m_referenceFrame );
	if( !m_Loader->open() )
	{
		std::stringstream stringstr;
		stringstr << "mesh '" << file_path << "' could not be loaded, please check it";
		throw drd::RuntimeError( grindGetRiObjectName() + stringstr.str() );
	}

	m_Loader->reportStats();

	const std::vector< Imath::V3f > & vertexVector = m_Loader->getVertexVector();
	const std::vector< Imath::V3f > & staticVertexVector = m_Loader->getStaticVertexVector();
	const std::vector< Imath::V3f > & normalVector = m_Loader->getNormalVector();
	const std::vector< Imath::V3f > & staticNormalVector = m_Loader->getStaticNormalVector();
    const std::vector< Imath::V2f > & texCoordVector = m_Loader->getTexCoordVector();
	const std::vector< bee::MeshPolygon > & polygonVector = m_Loader->getPolygonVector();

	bool containsNormal = !normalVector.empty();
	bool containsUV = !texCoordVector.empty();

	if( !containsNormal )
	{
		throw drd::RuntimeError( grindGetRiObjectName() + "mesh requires surface normal" );
	}

	size_t n = vertexVector.size();
	// surface normals will be averaged
	Imath::V3f up_vec( 0.0f, 1.0f, 0.0f );
	Imath::V3f zero_vec( 0.0f, 0.0f, 0.0f );

	m_N.resize( n, up_vec );
	m_StaticN.resize( n, up_vec );

	m_UV.assign( texCoordVector.begin(), texCoordVector.end() );

	n = polygonVector.size();

	// for each polygon
	for( size_t p = 0; p < n; ++p )
	{
		const bee::MeshPolygon& poly = polygonVector[p];
		int nv = poly.m_FaceVector.size();
		for( size_t v = 0; v < nv; ++v )
		{
			const bee::MeshFace& vert = poly.m_FaceVector[v];

			m_ConnectivityP.push_back( vert.vertexIdx );
			// note: we'll actually be using the same connectivity for N as for P as we're averaging
			m_ConnectivityN.push_back( vert.vertexIdx );
			m_ConnectivityUV.push_back( vert.texCoordIdx );
		}
		m_PolyVertCounts.push_back(nv);
	}

    GrabData( vertexVector, polygonVector, normalVector, m_P, m_N );
    GrabData( staticVertexVector, polygonVector, staticNormalVector, m_StaticP, m_StaticN );

	// update the bounding box
	calcBounds();
}

//-------------------------------------------------------------------------------------------------
void HostMesh::calcBounds()
{
	m_Bounds.populate( m_P.begin(), m_P.end(), 0 );
}

//-------------------------------------------------------------------------------------------------
bool HostMesh::setFrame( float a_Frame )
{
	DRD_LOG_ASSERT( L, m_Loader, "Not loaded, what's going on?" );
	if( !m_Loader->setFrame( a_Frame ) )
	{
		DRD_LOG_WARN( L, "Was NOT able to set frame to: " << a_Frame );
		return false;
	}

	const std::vector< Imath::V3f > & vertexVector = m_Loader->getVertexVector();
	const std::vector< Imath::V3f > & normalVector = m_Loader->getNormalVector();

	m_P.assign( vertexVector.begin(), vertexVector.end() );
	m_N.assign( normalVector.begin(), normalVector.end() );

	return true;
}

//-------------------------------------------------------------------------------------------------
void HostMesh::setAutoApplyTransforms( bool const a_autoApplyTransforms )
{
	if( m_Loader && ( m_Loader->getType() == bee::Loader::eVac ) )
	{
		( boost::dynamic_pointer_cast< bee::VacLoader >( m_Loader ) )->setAutoApplyTransforms( a_autoApplyTransforms );
	}

	m_autoApplyTransforms = a_autoApplyTransforms;
	DRD_LOG_DEBUG( L, "HostMesh::setAutoApplyTransforms = " << getAutoApplyTransforms() );
}

//-------------------------------------------------------------------------------------------------
bool const HostMesh::getAutoApplyTransforms() const
{
	if( m_Loader && ( m_Loader->getType() == bee::Loader::eVac ) )
	{
		return ( boost::dynamic_pointer_cast< bee::VacLoader >( m_Loader ) )->getAutoApplyTransforms();
	}

	return m_autoApplyTransforms;
}

//-------------------------------------------------------------------------------------------------
//! set the reference frame to be used for local transformations
void HostMesh::setReferenceFrame( float const a_frame )
{
	if( m_Loader && ( m_Loader->getType() == bee::Loader::eVac ) )
	{
		( boost::dynamic_pointer_cast< bee::VacLoader >( m_Loader ) )->setReferenceFrame( a_frame );
	}

	m_referenceFrame = a_frame;
	DRD_LOG_DEBUG( L, "HostMesh::setReferenceFrame = " << a_frame );
}

//-------------------------------------------------------------------------------------------------
DeviceMesh::DeviceMesh()
{
	init();
}

//-------------------------------------------------------------------------------------------------
DeviceMesh::DeviceMesh( const std::string& file_path )
{
	init();
	read( file_path );
}

//-------------------------------------------------------------------------------------------------
DeviceMesh::~DeviceMesh()
{
}

//-------------------------------------------------------------------------------------------------
void DeviceMesh::init()
{
	m_P.setBufferType( GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW );
	m_PDup.setBufferType( GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW );
	m_N.setBufferType( GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW );
	m_NDup.setBufferType( GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW );
	m_Tangent.setBufferType( GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW );
	m_VertIdDuplicate.setBufferType( GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW );
	m_UVDup.setBufferType( GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW );
	m_StaticP.setBufferType( GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW );
	m_StaticN.setBufferType( GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW );
	m_StaticTangent.setBufferType( GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW );
	m_CurrentFrame = 0.0f;
	m_PrevVertCount = 0;
	m_SubdIterations = 0;
	m_DisplayNormals = false;
	m_PDupDirty = true;
	m_NDupDirty = true;
	m_UVDupDirty = true;
	m_TangentsDirty = true;
	m_StaticTangentsDirty = true;

	// at this point we are not updated
	m_stateIsDirty = true;
}

//-------------------------------------------------------------------------------------------------
void DeviceMesh::setAutoApplyTransforms( bool const a_autoApplyTransforms )
{
	m_stateIsDirty = true;
	m_HostMesh.setAutoApplyTransforms( a_autoApplyTransforms );
}

//-------------------------------------------------------------------------------------------------
bool const DeviceMesh::getAutoApplyTransforms() const
{
	return m_HostMesh.getAutoApplyTransforms();
}

//-------------------------------------------------------------------------------------------------
//! set the reference frame to be used for local transformations
void DeviceMesh::setReferenceFrame( float const a_frame )
{
	m_stateIsDirty = true;
	m_HostMesh.setReferenceFrame( a_frame );
}

//-------------------------------------------------------------------------------------------------
void DeviceMesh::info() const
{
	std::cout << "DeviceMesh Info...\n";
	std::cout << "m_P.size(): " << m_P.size() << std::endl;
	std::cout << "m_StaticP.size(): " << m_StaticP.size() << std::endl;
}

//-------------------------------------------------------------------------------------------------
void DeviceMesh::read( const std::string& file_path )
{
	m_HostMesh.read( file_path );

    m_StaticP.setValue( m_HostMesh.m_StaticP );
    m_StaticN.setValue( m_HostMesh.m_StaticN );

    // update data in force mode, i.e. set the frame to 0, but this is arbitrary
    // since we are forcing the update to happen regardless
    setFrame( 0.0f );

	m_UV.setValue( m_HostMesh.m_UV );
	m_ConnectivityP.setValue( m_HostMesh.m_ConnectivityP );
	m_ConnectivityN.setValue( m_HostMesh.m_ConnectivityN );
	m_ConnectivityUV.setValue( m_HostMesh.m_ConnectivityUV );
	m_PolyVertCounts.setValue( m_HostMesh.m_PolyVertCounts );

	buildDuplicates();
	buildTangents();

	// state has been updated, don't need to update it now
	m_stateIsDirty = false;
}

//-------------------------------------------------------------------------------------------------
BBox DeviceMesh::getBounds() const
{
	// for now, recalculate on each request (could be cached with dirty flag etc)
	return grind::getBounds( m_P );
}

//-------------------------------------------------------------------------------------------------
void DeviceMesh::dumpGL( float lod ) const
{
	buildDuplicates();

	if ( m_PDup.size() != 0 )
	{
		glEnableClientState( GL_VERTEX_ARRAY );
		m_PDup.bindGL();
		glVertexPointer( 3, GL_FLOAT, 0, 0 );

		glEnableClientState( GL_NORMAL_ARRAY );
		m_NDup.bindGL();
		glNormalPointer( GL_FLOAT, 0, NULL );

		glEnableClientState( GL_TEXTURE_COORD_ARRAY );
		m_UVDup.bindGL();
		glTexCoordPointer( 2, GL_FLOAT, 0, NULL );

		glDrawArrays( GL_TRIANGLES, 0, m_PDup.size() );

		glDisableClientState( GL_TEXTURE_COORD_ARRAY );
		m_UVDup.unbindGL();

		glDisableClientState( GL_NORMAL_ARRAY );
		m_NDup.unbindGL();

		glDisableClientState( GL_VERTEX_ARRAY );
		m_PDup.unbindGL();
	}
	else if ( m_P.size() != 0 )
	{

		glEnableClientState( GL_VERTEX_ARRAY );
		m_P.bindGL();
		glVertexPointer( 3, GL_FLOAT, 0, 0 );

		glDrawArrays( GL_POINTS, 0, m_P.size() );

		glDisableClientState( GL_VERTEX_ARRAY );
		m_P.unbindGL();
	}
	else
	{
		DRD_LOG_INFO( L, "DeviceMesh::dumpGL can't find valid data to draw" );
	}
}

//----------------------------------------------------------------------------
void
DeviceMesh::drawNormals( float a_NormalSize )
{
	// totally unoptimal and not intended for frequent usage, more for debugging mesh problems
	if( m_PDup.size() != 0 && m_PDup.size() == m_NDup.size() )
	{
		HostVector< Imath::V3f > P, N;
		P.setValue( m_PDup );
		N.setValue( m_NDup );
		assert( P.size() == N.size() );
		int n = P.size();
		glBegin( GL_LINES );
		Imath::V3f PN;
		for( int i = 0; i < n; ++i ){
			glVertex3f( P[i].x, P[i].y, P[i].z );
			PN = P[i] + N[i]*a_NormalSize;
			glVertex3f( PN.x, PN.y, PN.z );
		}
		glEnd();
	}
}

//----------------------------------------------------------------------------
bool
DeviceMesh::setFrame( float a_Frame )
{
	// if we have not forced the update, then test for a previous invokation at this frame
	// else we force the update to occur regardless
	if( !m_stateIsDirty )
	{
		if( m_CurrentFrame == a_Frame )
		{
			return true;
		}
	}

	// state has been updated, don't need to update it now
	m_stateIsDirty = false;

	if( !m_HostMesh.setFrame( a_Frame ) )
	{
		DRD_LOG_ERROR( L, "Was NOT able to set frame to: " << a_Frame );
		return false;
	}

	m_P.setValue( m_HostMesh.m_P );
	m_N.setValue( m_HostMesh.m_N );
	m_CurrentFrame = a_Frame;

	setPointsAndNormalsChanged();

	return true;
}

//----------------------------------------------------------------------------
void DeviceMesh::setToStaticPose()
{
	m_P.setValue( m_StaticP );
	m_N.setValue( m_StaticN );
	m_CurrentFrame = 0.0f;
	setPointsAndNormalsChanged();
}

//----------------------------------------------------------------------------
void DeviceMesh::setP( const std::vector< Imath::V3f >& i_P )
{
	if( i_P.size() != m_P.size() ){
		std::ostringstream ss;
		ss << "tried to set P on a mesh with incorrect number of points (" << i_P.size() << " provided, mesh has " << m_P.size() << ")";
		throw drd::RuntimeError( grindGetRiObjectName() + ss.str() );
	}
	m_P.setValue( i_P );
	setPointsAndNormalsChanged();
}

//----------------------------------------------------------------------------
void DeviceMesh::setN( const std::vector< Imath::V3f >& i_N )
{
	if( i_N.size() != m_N.size() ){
		std::ostringstream ss;
		ss << "tried to set N on a mesh with incorrect number of normals (" << i_N.size() << " provided, mesh has " << m_N.size() << ")";
		throw drd::RuntimeError( grindGetRiObjectName() + ss.str() );
	}
	m_N.setValue( i_N );
	setPointsAndNormalsChanged();
}

//----------------------------------------------------------------------------
void DeviceMesh::buildDuplicates() const
{
	if( !(m_PDupDirty || m_NDupDirty || m_UVDupDirty ) )
		return;

	DeviceVector<int> cumulative_vert_counts;
	cumulative_vert_counts.resize( m_PolyVertCounts.size(), 0 );
	inclusiveScan( m_PolyVertCounts, cumulative_vert_counts );

	DeviceVector<int> cumulative_tri_counts;
	cumulative_tri_counts.resize( m_PolyVertCounts.size() );
	int duplicate_array_size = calcCumulativeTriCountDevice( m_PolyVertCounts.size(), m_PolyVertCounts.getDevicePtr(), cumulative_tri_counts.getDevicePtr() );

	if( m_PDupDirty )
	{
		assert( m_P.size() > 0 );
		DeviceVector<int>& p_duplicate_map = m_VertIdDuplicate;
		p_duplicate_map.resize( duplicate_array_size, 0 );
		buildTriConnectivityDevice( cumulative_vert_counts.size(), cumulative_vert_counts.getDevicePtr(), m_ConnectivityP.getDevicePtr(), cumulative_tri_counts.getDevicePtr(), p_duplicate_map.getDevicePtr() );

		DeviceVector<Imath::V3f>& p_dup = m_PDup;
		p_dup.resize( duplicate_array_size );
		buildDuplicatesDevice( duplicate_array_size, p_duplicate_map.getDevicePtr(), getP().getDevicePtr(), p_dup.getDevicePtr() );

		m_PDupDirty = false;
	}

	if( m_NDupDirty )
	{
		assert( m_N.size() > 0 );
		DeviceVector<int> n_duplicate_map;
		n_duplicate_map.resize( duplicate_array_size, 0 );
		buildTriConnectivityDevice( cumulative_vert_counts.size(), cumulative_vert_counts.getDevicePtr(), m_ConnectivityN.getDevicePtr(), cumulative_tri_counts.getDevicePtr(), n_duplicate_map.getDevicePtr() );

		DeviceVector<Imath::V3f>& n_dup = m_NDup;
		n_dup.resize( duplicate_array_size );
		buildDuplicatesDevice( duplicate_array_size, n_duplicate_map.getDevicePtr(), getN().getDevicePtr(), n_dup.getDevicePtr() );

		m_NDupDirty = false;
	}

	if( m_UVDupDirty )
	{
		assert( m_UV.size() > 0 );
		DeviceVector<int> uv_duplicate_map;
		uv_duplicate_map.resize( duplicate_array_size, 0 );
		buildTriConnectivityDevice( cumulative_vert_counts.size(), cumulative_vert_counts.getDevicePtr(), m_ConnectivityUV.getDevicePtr(), cumulative_tri_counts.getDevicePtr(), uv_duplicate_map.getDevicePtr() );

		DeviceVector<Imath::V2f>& uv_dup = m_UVDup;
		uv_dup.resize( duplicate_array_size );
		buildDuplicatesDevice( duplicate_array_size, uv_duplicate_map.getDevicePtr(), getUV().getDevicePtr(), uv_dup.getDevicePtr() );

		m_UVDupDirty = false;
	}
}

//----------------------------------------------------------------------------
const grind::DeviceVector<Imath::V3f>& DeviceMesh::getPDuplicate() const
{
	buildDuplicates();
	return m_PDup;
}

//----------------------------------------------------------------------------
const grind::DeviceVector<Imath::V3f>& DeviceMesh::getNDuplicate() const
{
	buildDuplicates();
	return m_NDup;
}

//----------------------------------------------------------------------------
const grind::DeviceVector<Imath::V2f>& DeviceMesh::getUVDuplicate() const
{
	buildDuplicates();
	return m_UVDup;
}

const DeviceVector<int>& DeviceMesh::getVertIdDuplicate() const
{
	buildDuplicates();
	return m_VertIdDuplicate;
}

//----------------------------------------------------------------------------
const grind::DeviceVector<Imath::V3f>& DeviceMesh::getTangent() const
{
	buildTangents();
	return m_Tangent;
}

//----------------------------------------------------------------------------
const grind::DeviceVector<Imath::V3f>& DeviceMesh::getStaticTangent() const
{
	buildTangents();
	return m_StaticTangent;
}

//----------------------------------------------------------------------------
void DeviceMesh::setTopologyChanged()
{
	m_PrevVertCount = 0;
	m_UVDupDirty = true;
	m_StaticTangentsDirty = true;

	setPointsAndNormalsChanged();
}

//----------------------------------------------------------------------------
void DeviceMesh::setPointsAndNormalsChanged()
{
	m_PDupDirty = true;
	m_NDupDirty = true;
	m_TangentsDirty = true;
}

//----------------------------------------------------------------------------
void DeviceMesh::buildTangents() const
{
	if( !(m_TangentsDirty || m_StaticTangentsDirty) ) return;

	// make sure we're not operating on an empty mesh
	assert( m_P.size() != 0 );
	int vert_count = m_P.size();
	bool topology_changed = m_PrevVertCount != vert_count;
	m_PrevVertCount = vert_count;

	if( topology_changed || m_TangentVertId.size() == 0 ){
		// currently host code since should be calculated infrequently and is difficult to parallelize
		DRD_LOG_INFO( L, "recomputing tangent topology" );
		HostVector<Imath::V3f> h_P, h_T;
		HostVector<int> h_PolyVertCounts, h_ConnectivityP, h_TangentId;
		h_P.setValue( m_P );
		h_TangentId.resize( m_P.size(), 0 );
		h_PolyVertCounts.setValue( m_PolyVertCounts );
		h_ConnectivityP.setValue( m_ConnectivityP );
		h_T.resize( h_P.size() );

		// for each face
		for( int f = 0, c = 0; f < h_PolyVertCounts.size(); ++f ){
			int c0 = c;
			int face_vert_count = h_PolyVertCounts[f];
			for( int v = 0; v < face_vert_count; ++v, ++c ){
				int this_cv = h_ConnectivityP[ c0 + v ];
				int next_cv = h_ConnectivityP[ c0 + ((v+1)%face_vert_count) ];
				// for each vert, store a connect vert id (may overwrite prev values)
				h_TangentId[ this_cv ] = next_cv;
			}
		}
		m_TangentVertId.setValue( h_TangentId );
	}

	if( m_TangentsDirty ){
		m_Tangent.resize( vert_count );
		computeTangentsDevice( vert_count, m_TangentVertId.getDevicePtr(), m_P.getDevicePtr(), m_N.getDevicePtr(), m_Tangent.getDevicePtr() );
		m_TangentsDirty = false;
	}

	if( m_StaticTangentsDirty ){
		m_StaticTangent.resize( vert_count );
		computeTangentsDevice( vert_count, m_TangentVertId.getDevicePtr(), m_StaticP.getDevicePtr(),  m_StaticN.getDevicePtr(), m_StaticTangent.getDevicePtr() );
		m_StaticTangentsDirty = false;
	}
}


void DeviceMesh::saveData( const std::string& key, const std::string& path ) const
{
	if( key == "P" )
		save( m_P, path );
	else if( key == "N" )
		save( m_N, path );
	else if( key == "UV" )
		save( m_UV, path );
	else if( key == "staticP" )
		save( m_StaticP, path );
	else if( key == "staticN" )
		save( m_StaticN, path );
	else
		throw std::runtime_error( "unsupported key" );
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
