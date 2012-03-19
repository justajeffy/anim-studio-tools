/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: $"
 * SVN_META_ID = "$Id: line_soup.h 102193 2011-09-12 04:02:29Z luke.emrose $"
 */

#ifndef grind_line_soup_h
#define grind_line_soup_h

//-------------------------------------------------------------------------------------------------
#include "vector_set.h"
#include "device_vector.h"
#include "host_vector.h"
#include "renderable.h"

#include <vector>
#include <string>

namespace grind
{

#define LINESOUP_PARAM_GEO_TYPE "geo_type"
#define LINESOUP_PARAM_ALIGN_MODE "align_mode"
#define LINESOUP_PARAM_WIDTH_MODE "width_mode"
#define LINESOUP_PARAM_INTERPOLATION_MODE "interpolation_mode"

//-------------------------------------------------------------------------------------------------
//! general display container drawing line segments with sorting capabilities
struct LineSoup: public Renderable
{
	//! different available types for GL rendering
	enum GeoType { LINESOUP_GL_LINES, LINESOUP_GL_QUADS };

	//! different alignment modes for width of curve
	enum AlignMode { LINESOUP_CAMERA_FACING, LINESOUP_NORMAL_FACING };

	//! different interpolation modes supported
	enum InterpolationMode { LINESOUP_LINEAR, LINESOUP_CATMULL_ROM };

	//! different width modes supported
	enum WidthMode { LINESOUP_CONSTANT_WIDTH, LINESOUP_VARYING_WIDTH };

	//! default constructor
	LineSoup();

	//! set line width
	void setConstantLineWidth( float i_Width ){ m_ConstantWidth = i_Width; }

	//! @cond DEV

	//~ destructor
	~LineSoup();

	//! clear out data for container re-use
	void clear();

	//! sort based on eye position
	void viewSort() const;

	//! initialize the data to be an x-y-z axis (useful for testing)
	void testSetup();

	//! const access to points
	const DeviceVector< Imath::V3f >& getP() const{ return m_P; }

	//! access points
	DeviceVector< Imath::V3f >& getP();

	//! access nordrd::mals
	DeviceVector< Imath::V3f >& getN(){ return m_N; }

	//! access UVW coordinates
	DeviceVector< Imath::V4f >& getUVW(){ return m_UVW; }

	//! access lod rand
	DeviceVector< float >& getLodRand() { return m_LodRand; }

	//! access varying width param
	DeviceVector< float >& getWidth() { return m_Width; }

	//! get the number of cvs per curve
	unsigned int getCurveCount() const { return m_CurveCount; }

	//! get the number of cvs per curve
	unsigned int getCvCount() const { return m_CvCount; }

	//! list the available parameters that can be set via setParam()
	std::vector< std::string > listParams();

	//! set various params
	void setParam( const std::string& param, const std::string& val );

	//! resize data
	void resize( unsigned int n_curves, unsigned int n_cvs );

	//! set the max lod so it can be exposed via renderman etc
	void setLod( float i_Lod ) { m_Lod = i_Lod; }

	//! flag that you have finished writing vertex/normal data to the linesoup
	void finalizePN();

	//! flag that you have finished writing uvw data to the linesoup
	void finalizeUVW();

	//! return the geo type as set by the user
	GeoType getUserGeoType() const { return m_GeoType; }

	//! return the actual geo type of the underlying data
	GeoType getActualGeoType() const;

	//! return the alignment mode used by this instance
	AlignMode getAlignMode() const;

	BBox getBounds() const;

	//! access the prim var arrays
	HostVectorSet& getPrimVars() { return m_PrimVars; }

	//! set the string specifying variable type that will be injected into the rib
	void setPrimVarType( const std::string& i_VarName, const std::string& i_VarType )
	{ m_PrimVarTypes[ i_VarName ] = i_VarType; }

	void setDisplayNormals( bool val ){ m_DisplayNormals = val; }
	bool getDisplayNormals() const { return m_DisplayNormals; }

	//! python access to data
	void getData( const std::string& name, std::vector< Imath::V3f >& result ) const;
	void setData( const std::string& name, const std::vector< Imath::V3f >& src );

	//! log some info to std::out
	void info() const;

private:

	void dumpGL( float lod ) const;
	void dumpRib( float rib ) const;

	bool m_Initialized;

	int m_CurveCount;
	int m_CvCount;

	DeviceVector< Imath::V3f > m_P;
	DeviceVector< Imath::V3f > m_N;
	DeviceVector< Imath::V4f > m_UVW;
	DeviceVector< Imath::V3f > m_Colour;
	DeviceVector< float > m_LodRand;
	DeviceVector< float > m_Width;
	mutable DeviceVector< unsigned int > m_IndicesGL;

	//! the set of primitive vars (only float/Imath::V3f supported)
	HostVectorSet m_PrimVars;

	//! the types of the various primvars
	std::map< std::string, std::string > m_PrimVarTypes;

	//! key used in sorting (distance from eye pos)
	mutable DeviceVector< float > m_SortKeys;

	//! value used in sorting (line segment index)
	mutable DeviceVector< unsigned int > m_SortValues;

	//! width of lines when rendering with constant width
	float m_ConstantWidth;

	//! flag when P has been modified
	bool m_DirtyP;

	//! the current curve display type for OpenGL
	GeoType m_GeoType;

	//! the current alignment mode
	AlignMode m_AlignMode;

	//! the current width mode
	WidthMode m_WidthMode;

	//! the current interpolation mode
	InterpolationMode m_InterpolationMode;

	//! user param names list
	std::vector< std::string > m_UserParamNames;

	//! initialize based on curve count and number of cvs
	void initIndices();

	//! store the max lod
	float m_Lod;

	bool m_DisplayNormals;

	//! @endcond
};

}

#endif /* grind_line_soup_h */


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
