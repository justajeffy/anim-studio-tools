/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: $"
 * SVN_META_ID = "$Id: guide_set.h 96578 2011-08-15 06:23:56Z luke.emrose $"
 */

#ifndef grind_guide_set_h
#define grind_guide_set_h

//-------------------------------------------------------------------------------------------------
#include <boost/shared_ptr.hpp>
#include "grind/device_vector.h"
#include "grind/renderable.h"
#include "grind/guide_params.h"

// forward declaration
namespace bee
{
	class MeshLoader;
}

namespace grind
{

//-------------------------------------------------------------------------------------------------
class DeviceMesh;

//-------------------------------------------------------------------------------------------------
//! dynamic per-vertex guide curves
class GuideSet
: public Renderable
{
public:
	//! default constructor
	GuideSet();

	//! initialize with some basic values
	void init( 	unsigned int n_curves,
				int n_cvs,
				float guide_length );

	//! make guides stick straight out along surface mesh normals (used for testing)
	void surfaceNormalGroom( const DeviceMesh& mesh );

	//! read in an obj file from disk and snapshot guides into tangent space of supplied mesh
	void read( const std::string& i_Path, const DeviceMesh& mesh );

	//! update based on a current mesh position and tangent space guides
	void update( const DeviceMesh& mesh );

	//! @cond DEV
	bool setFrame( float a_Frame, const DeviceMesh& a_Mesh );

	//! resize data
	void resize( unsigned int n_curves, unsigned int n_cvs );

	//! fill up with some random data
	void randomize( unsigned int n_curves, unsigned int n_cvs );

	//! set up the indices for drawing line segments
	void setupIndices();

	//! snapshot into tangent space
	void snapshotIntoTangentSpace( const DeviceMesh& mesh, const HostVector<Imath::V3f>& guide_verts );

	unsigned int getNCurves() const { return m_CurveCount; }
	void setNCurves( unsigned int i_CurveCount ) { m_CurveCount = i_CurveCount; }
	unsigned int getNCVs() const { return m_CVCount; }
	void setNCVs( unsigned int i_CVCount ) { m_CVCount = i_CVCount; }

	const DeviceVector< Imath::V3f >& getP() const { return m_P; }
	DeviceVector< Imath::V3f >& getP() { return m_P; }

	const DeviceVector< Imath::V3f >& getAcross() const { return m_Across; }
	DeviceVector< Imath::V3f >& getAcross() { return m_Across; }

	BBox getBounds() const;

	//! access to guide length
	const DeviceVector< float >& getGuideLength() const;

	//! the max element of getGuideLength()
	float getMaxGuideLength() const;

	//! python access to data
	void getData( const std::string& name, std::vector< Imath::V3f >& result ) const;
	void setData( const std::string& name, const std::vector< Imath::V3f >& src );

private:

	//! dump the appropriate OpenGL calls
	void dumpGL( float lod ) const;
	//! dump the appropriate RIB calls
	void dumpRib( float lod ) const;

	//! number of guide curves
	unsigned int m_CurveCount;

	//! number of cvs per curve
	unsigned int m_CVCount;

	//! current position
	DeviceVector< Imath::V3f > m_P;

	//! current position in tangent space
	DeviceVector< Imath::V3f > m_PTangent;

	//! across vector in object space
	DeviceVector< Imath::V3f > m_Across;

	//! across vector in tangent space
	DeviceVector< Imath::V3f > m_AcrossTangent;

	//! across vector (display only)
	mutable DeviceVector< Imath::V3f > m_AcrossDisplay;

	//! previous position for verlet integration
	DeviceVector< Imath::V3f > m_PrevP;

	//! per-guide span length cache
	DeviceVector< float > m_SpanLength;

	//! indices used for drawing as line segments
	DeviceVector< unsigned int > m_Indices;

	//! per guide dynamic parameters
	DeviceVector< GuideParams > m_Params;

	//! per guide length variable
	mutable DeviceVector< float > m_GuideLength;

	//! calculate the guide length of guide curves
	void calcGuideLength() const;

	//! keep around the loader so that weak pointers in cache are not the only handles
	boost::shared_ptr< const bee::MeshLoader > m_Loader;

	//! keep the file path around so we can switch to cached loaders
	std::string m_FilePath;

	//! @endcond
};

} // grind

#endif /*grind_guide_set_h*/



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
