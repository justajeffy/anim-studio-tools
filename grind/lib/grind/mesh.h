/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: $"
 * SVN_META_ID = "$Id: mesh.h 85128 2011-06-02 04:40:02Z luke.emrose $"
 */

#ifndef grind_mesh_h
#define grind_mesh_h

#include <boost/shared_ptr.hpp>

//-------------------------------------------------------------------------------------------------
#include "bbox.h"
#include "host_vector.h"
#include "device_vector.h"
#include "renderable.h"

//-------------------------------------------------------------------------------------------------
// forward declaration
namespace bee
{
	class MeshLoader;
}

namespace grind
{

//-------------------------------------------------------------------------------------------------
//! representation of a mesh stored on host
class HostMesh
{
public:
	//! default constructor
	HostMesh();

	//! read from a file path
	void read( const std::string& file_path );

	//! @cond DEV
	bool setFrame( float a_Frame );

	size_t nVerts() const { return m_P.size(); };

	//! calculate the bounding box
	void calcBounds();

	//! access the bounding box
	grind::BBox getBounds() const { return m_Bounds; }

	//! set the vac automatic transform application
	void setAutoApplyTransforms( bool const a_autoApplyTransforms );
	bool const getAutoApplyTransforms() const;
	//! set the reference frame to be used for local transformations
	void setReferenceFrame( float const a_frame );

	//! the bounding box
	grind::BBox m_Bounds;

	grind::HostVector<Imath::V3f> m_P;
	grind::HostVector<Imath::V3f> m_N;
	grind::HostVector<Imath::V2f> m_UV;

	grind::HostVector<Imath::V3f> m_StaticP;
	grind::HostVector<Imath::V3f> m_StaticN;

	//! keep around the loader for getting frame updates
	boost::shared_ptr< bee::MeshLoader > m_Loader;

	//! polygon info
	grind::HostVector<int> m_ConnectivityP;
	grind::HostVector<int> m_ConnectivityN;
	grind::HostVector<int> m_ConnectivityUV;
	grind::HostVector<int> m_PolyVertCounts;

	bool m_autoApplyTransforms;
	float m_referenceFrame;

	//! @endcond
};


//-------------------------------------------------------------------------------------------------
//! representation of a mesh stored on device
class DeviceMesh : public Renderable
{
public:
	//! default constructor
	DeviceMesh();
	virtual ~DeviceMesh();

	//! construct from a file path
	DeviceMesh( const std::string& file_path );

	//! shared initialization shared between constructors
	void init();

	//! dump some info to std::cout
	void info() const;

	//! getBounds() from Renderable
	virtual BBox getBounds() const;

	//! read from a file path
	void read( const std::string & file_path );

	size_t nVerts() const { return m_P.size(); };

	//! set to a frame in the Vac
	bool setFrame( float a_Frame );

	//! set to rest pose
	void setToStaticPose();

	// flag that topology changed
	void setTopologyChanged();
	void setPointsAndNormalsChanged();

	//! get the subdivision level (0 if the mesh hasn't been subdivided)
	int getSubdIterations() const { return m_SubdIterations; }

	//! set the subd iterations
	void setSubdIterations( int iter ) { m_SubdIterations = iter; }

	void drawNormals( float a_NormalSize );

	void setP( const std::vector< Imath::V3f >& i_P );
	void setN( const std::vector< Imath::V3f >& i_P );

	//! @cond DEV

	grind::DeviceVector<Imath::V3f>& getP() { return m_P; }
	const grind::DeviceVector<Imath::V3f>& getP() const { return m_P; }

	grind::DeviceVector<Imath::V3f>& getN() { return m_N; }
	const grind::DeviceVector<Imath::V3f>& getN() const { return m_N; }

	grind::DeviceVector<Imath::V2f>& getUV() { return m_UV; }
	const grind::DeviceVector<Imath::V2f>& getUV() const { return m_UV; }

	const grind::DeviceVector<Imath::V3f>& getStaticP() const { return m_StaticP; }
	const grind::DeviceVector<Imath::V3f>& getStaticN() const { return m_StaticN; }

	const grind::DeviceVector<int>& getConnectivityP() const { return m_ConnectivityP; }
	const grind::DeviceVector<int>& getConnectivityN() const { return m_ConnectivityN; }
	const grind::DeviceVector<int>& getConnectivityUV() const { return m_ConnectivityUV; }
	const grind::DeviceVector<int>& getPolyVertCounts() const { return m_PolyVertCounts; }

	grind::DeviceVector<int>& getConnectivityP() { return m_ConnectivityP; }
	grind::DeviceVector<int>& getConnectivityN() { return m_ConnectivityN; }
	grind::DeviceVector<int>& getConnectivityUV() { return m_ConnectivityUV; }
	grind::DeviceVector<int>& getPolyVertCounts() { return m_PolyVertCounts; }

	void setDisplayNormals( bool val ){ m_DisplayNormals = val; }
	bool getDisplayNormals() const { return m_DisplayNormals; }

	const grind::DeviceVector<int>& getVertIdDuplicate() const;
	const grind::DeviceVector<Imath::V3f>& getPDuplicate() const;
	const grind::DeviceVector<Imath::V3f>& getNDuplicate() const;
	const grind::DeviceVector<Imath::V2f>& getUVDuplicate() const;
	const grind::DeviceVector<Imath::V3f>& getTangent() const;
	const grind::DeviceVector<Imath::V3f>& getStaticTangent() const;

	//! set the vac automatic transform application
	void setAutoApplyTransforms( bool const a_autoApplyTransforms );
	bool const getAutoApplyTransforms() const;
	//! set the reference frame to be used for local transformations
	void setReferenceFrame( float const a_frame );

	void saveData( const std::string& key, const std::string& path ) const;

private:

	//! build the duplicate data (used for GL drawing etc)
	void buildDuplicates() const;

	void buildTangents() const;

	void dumpGL( float lod ) const;

	float m_CurrentFrame;

	grind::DeviceVector< Imath::V3f > m_P;
	grind::DeviceVector< Imath::V3f > m_N;
	grind::DeviceVector< Imath::V2f > m_UV;

	grind::DeviceVector< Imath::V3f > m_StaticP;
	grind::DeviceVector< Imath::V3f > m_StaticN;

	// aux data that will be generated when needed
	mutable grind::DeviceVector< Imath::V3f > m_PDup;
	mutable grind::DeviceVector< Imath::V3f > m_NDup;
	mutable grind::DeviceVector< Imath::V2f > m_UVDup;
	mutable grind::DeviceVector< Imath::V3f > m_Tangent;
	mutable grind::DeviceVector< int > m_VertIdDuplicate;
	mutable grind::DeviceVector< Imath::V3f > m_StaticTangent;

	mutable bool m_PDupDirty;
	mutable bool m_NDupDirty;
	mutable bool m_UVDupDirty;
	mutable bool m_TangentsDirty;
	mutable bool m_StaticTangentsDirty;
	//! detect topology changes
	mutable size_t m_PrevVertCount;

	//! I need to keep it around so that I don't have to re-load stuff
	HostMesh m_HostMesh;

	//! polygon info
	grind::DeviceVector<int> m_ConnectivityP;
	grind::DeviceVector<int> m_ConnectivityN;
	grind::DeviceVector<int> m_ConnectivityUV;
	grind::DeviceVector<int> m_PolyVertCounts;

	//! for each vert id, store an adjacent vert id for tangent calculations
	mutable grind::DeviceVector<int> m_TangentVertId;

	//! keep track of the subd iterations
	int m_SubdIterations;

	bool m_DisplayNormals;

	bool m_stateIsDirty;
	//! @endcond
};

} // namespace grind

#endif /* grind_mesh_h */


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
