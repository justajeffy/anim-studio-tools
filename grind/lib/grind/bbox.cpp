/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: $"
 * SVN_META_ID = "$Id: bbox.cpp 99443 2011-08-30 01:14:50Z hugh.rayner $"
 */

//-------------------------------------------------------------------------------------------------

#include <drdDebug/log.h>
DRD_MKLOGGER( L, "drd.grind.BBox" );

#include "bbox.h"
#include "log.h"
#include "random.h"
#include <bee/gl/ScopeHelpers.h>
#include <limits>

#include "GL/gl.h"
#include <ri.h>
#include <boost/functional/hash.hpp>

#define BBOX_OPACITY 0.2

//-------------------------------------------------------------------------------------------------
using namespace grind;

/*static*/ int BBox::s_BBOX_MODE = 0;

//-------------------------------------------------------------------------------------------------
void BBox::buildArrays( std::vector< VecType > & result ) const
{
	result.clear();

	const VecType & min = GetBox().min;
	const VecType & max = GetBox().max;
	const MatType & mat = GetTransform();

	result.push_back( VecType( max.x, max.y, max.z ) );
	result.push_back( VecType( min.x, max.y, max.z ) );
	result.push_back( VecType( min.x, min.y, max.z ) );
	result.push_back( VecType( max.x, min.y, max.z ) );

	result.push_back( VecType( max.x, max.y, max.z ) );
	result.push_back( VecType( max.x, min.y, max.z ) );
	result.push_back( VecType( max.x, min.y, min.z ) );
	result.push_back( VecType( max.x, max.y, min.z ) );

	result.push_back( VecType( max.x, max.y, max.z ) );
	result.push_back( VecType( max.x, max.y, min.z ) );
	result.push_back( VecType( min.x, max.y, min.z ) );
	result.push_back( VecType( min.x, max.y, max.z ) );

	result.push_back( VecType( min.x, max.y, max.z ) );
	result.push_back( VecType( min.x, max.y, min.z ) );
	result.push_back( VecType( min.x, min.y, min.z ) );
	result.push_back( VecType( min.x, min.y, max.z ) );

	result.push_back( VecType( min.x, min.y, min.z ) );
	result.push_back( VecType( max.x, min.y, min.z ) );
	result.push_back( VecType( max.x, min.y, max.z ) );
	result.push_back( VecType( min.x, min.y, max.z ) );

	result.push_back( VecType( max.x, min.y, min.z ) );
	result.push_back( VecType( min.x, min.y, min.z ) );
	result.push_back( VecType( min.x, max.y, min.z ) );
	result.push_back( VecType( max.x, max.y, min.z ) );

	for( std::vector< VecType >::iterator it = result.begin(); it != result.end(); ++it )
	{
		mat.multVecMatrix( *it, *it );
	}
}

//-------------------------------------------------------------------------------------------------
void makeColourFromIndex( int id, float* c )
{
	unsigned int hashr = boost::hash<unsigned int>()(static_cast<unsigned int>(id) * 5 + 211 );
	unsigned int hashg = boost::hash<unsigned int>()(static_cast<unsigned int>(id) * 17 + 359 );
	unsigned int hashb = boost::hash<unsigned int>()(static_cast<unsigned int>(id) * 23 + 7919 );

	c[0] = (hashr % 256) / 255.0;
	c[1] = (hashg % 256) / 255.0;
	c[2] = (hashb % 256) / 255.0;

	c[3] = BBOX_OPACITY;
}

//-------------------------------------------------------------------------------------------------
void makeColourFromUInt( unsigned int col, float* c )
{
	unsigned char r = (col & 0xFF) >> 0;
	unsigned char g = (col & 0xFF00) >> 8;
	unsigned char b = (col & 0xFF0000) >> 16;

	float f = 1.f / 255.f;
	c[0] = b * f;
	c[1] = g * f;
	c[2] = r * f;

	c[3] = BBOX_OPACITY;
}

//-------------------------------------------------------------------------------------------------
void BBox::dumpGL( float lod ) const
{
	std::vector< VecType > p;
	buildArrays( p );
	float c[4];
	if (m_Colour != 0) makeColourFromUInt( m_Colour, c );
	else makeColourFromIndex( m_ColourIndex, c );

	bee::glDepthMaskHelper dmh( GL_FALSE );
	bee::glEnableHelper en0( GL_BLEND );
	bee::glEnableHelper en1( GL_DEPTH_TEST );
	bee::glBlendFuncHelper bf( GL_SRC_ALPHA, GL_ONE );

    SAFE_GL( glShadeModel( GL_SMOOTH ) );

    SAFE_GL( glEnableClientState( GL_VERTEX_ARRAY) );
	SAFE_GL( glVertexPointer( 3, GL_FLOAT, 0, &p[0] ) );

	SAFE_GL( glColor4f( c[0], c[1], c[2], c[3] ) );

	SAFE_GL( glDrawArrays( GL_QUADS, 0, p.size() ) );

	SAFE_GL( glDisableClientState( GL_VERTEX_ARRAY ) );
}

//-------------------------------------------------------------------------------------------------
void BBox::dumpRib( float lod ) const
{
	std::vector< VecType > p;
	buildArrays( p );
	float c[4];
	makeColourFromIndex( m_ColourIndex, c );

	VecType col( c[0], c[1], c[2] );
	std::vector< VecType > cs( p.size(), col );

	VecType opacity( c[3], c[3], c[3] );
	std::vector< VecType > os( p.size(), opacity );

	RtInt nverts[] =
	{ 4, 4, 4, 4, 4, 4 };

	RtInt verts[] =
	{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 };

	RiAttributeBegin();
	RiSides( 2 ); // make sure orientation issues don't bite us
	RiDisplacement( "null", RI_NULL ); // make sure displacement doesn't apply
	RiSurface( "defaultsurface", RI_NULL ); // can't use plastic or drd light shaders may explode
	RiPointsPolygons( 6, nverts, verts, "P", &p[ 0 ], "constant color Cs", &cs[ 0 ], "constant color Os", &os[ 0 ], RI_NULL );
	RiAttributeEnd();
}

//-------------------------------------------------------------------------------------------------
BBox::BBox()
: OrientedBoundingBox<float>()
, m_ColourIndex(0)
{}


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
