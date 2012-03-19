#ifndef _NAPALM_DELIGHT_GEOPROCBASE__H_
#define _NAPALM_DELIGHT_GEOPROCBASE__H_

#include <napalm/core/Table.h>
#include <GxWrappers.h>

namespace napalm_delight
{

//-------------------------------------------------------------------------------------------------
//! Base class for procedurals that emit via GX
class GeoProcBase
{
public:

	enum DefaultRecursionLevels
	{
		PRIMITIVE_PROC, //!< per primitive proc emission
		FACE_PROC, //!< per face proc emission
		GEO_PROC //!< per face geometry emission
	};

	typedef int RecursionLevel;

	GeoProcBase( const RecursionLevel rl );
	GeoProcBase( const GeoProcBase& src, const RecursionLevel rl, int faceId );
	virtual ~GeoProcBase();

	// client to provide these
	virtual void parse( const std::string& paramstr );
	virtual void emitPrimitiveProc( float detailSize ) = 0;
	virtual void emitFaceProcs( float detailSize ) = 0;
	virtual void emitGeo( float detailSize ) = 0;

	virtual float getCrudeBoundPad() const;

	//! this applies crudeBoundPad() to the primitive bound
	void getPrimitiveBound( RtBound result );

	//! access the 3delight Gx handle
	::GxGeometryHandle getRawGeoHandle();

	void setUpGeoHandle();

	virtual void subdivide( RtFloat detailSize );

	// gx surface evaluation
	bool evalSurf( const drd::GxSurfacePoints& pts, const char* name, std::vector<Imath::V3f>& result );
	bool evalSurf( const drd::GxSurfacePoints& pts, const char* name, std::vector<float>& result );
	bool evalSurf( const drd::GxSurfacePoints& pts, const char* name, napalm::V3fBufferPtr result );
	bool evalSurf( const drd::GxSurfacePoints& pts, const char* name, napalm::FloatBufferPtr result );

	Imath::V3f getCameraPos();

	//!
	void optionallyRenderPrimitiveBound( RtBound bound, bool force = false );
	void optionallyRenderFaceBound( RtBound bound, bool force = false );

	//! if you just want to render some bound (use the previous in preference )
	void renderBound( RtBound );

	//! calculate the length of a curve (will use Pref if available)
	float calcCurveLength( int samples = 3 );

	//! allow returning a list of sub-face areas
	float calcSubFaceAreas( unsigned int i_faceId, int nu, int nv, std::vector< float >* areas = NULL );

	//! calculate the area of a face (will use Pref if available)
	float calcFaceArea( unsigned int i_faceId );


	//-----------------------------------
	// Motion blur related
	//-----------------------------------

	//! emit an appropriate RiMotionBeginV() call if required
	void emitMotionBegin() const;

	//! emit a RiMotionEnd() if required
	void emitMotionEnd() const;

	//! access to time samples to pass into Gx etc
	const std::vector<float>& getGxTimeSamples() const;

	//! access to time samples used for output
	const std::vector<float>& getOutputTimeSamples() const;

protected:

	napalm::object_table_ptr m_userParams;

	drd::GxGeometryHandle m_geoHandle;
	RecursionLevel m_recursionLevel;
	int m_faceId;
	unsigned int m_faceCount;

private:
	//! don't allow copy constructor
	GeoProcBase( GeoProcBase& src );

	void setUpMotionBlurData();

	// stored outside of m_userParams for simpler handling
	std::vector< float > m_timeSamples;
	std::vector< float > m_offsetTimeSamples;
};


//-------------------------------------------------------------------------------------------------
#define STANDARD_GEOPROCBASE_PREDEFINES() \
extern "C" \
{ \
RtPointer DLLEXPORT ConvertParameters( RtString paramstr ); \
RtVoid DLLEXPORT Subdivide( RtPointer data, RtFloat detailSize ); \
RtVoid DLLEXPORT Free( RtPointer data ); \
}


//-------------------------------------------------------------------------------------------------
#define STANDARD_GEOPROCBASE_ENTRY( CLIENT_TYPE, RECURSION_LEVEL ) \
extern "C" \
{ \
RtPointer DLLEXPORT ConvertParameters( RtString paramstr ) \
{ \
	CLIENT_TYPE* my_data = new CLIENT_TYPE( RECURSION_LEVEL ); \
\
	my_data->parse( paramstr ); \
	my_data->setUpGeoHandle(); \
\
	return (RtPointer) my_data; \
} \
\
RtVoid DLLEXPORT Subdivide( RtPointer data, \
							RtFloat detailSize ) \
{ \
	CLIENT_TYPE* my_data = reinterpret_cast< CLIENT_TYPE* > ( data ); \
\
	my_data->subdivide( detailSize ); \
} \
\
RtVoid DLLEXPORT Free( RtPointer data ) \
{ \
	CLIENT_TYPE* my_data = reinterpret_cast< CLIENT_TYPE* > ( data ); \
	delete my_data; \
} \
}


}

#endif


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
