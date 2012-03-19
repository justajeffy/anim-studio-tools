/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: $"
 * SVN_META_ID = "$Id: cube.cpp 45312 2010-09-09 06:11:02Z chris.cooper $"
 */

//-------------------------------------------------------------------------------------------------
#include "log.h"
DRD_MKLOGGER( L, "drd.grind.Cube" );

#include "cube.h"
#include "random.h"

// gl includes
#include <GL/gl.h>

// renderman includes
#include <ri.h>

//-------------------------------------------------------------------------------------------------
using namespace grind;

//-------------------------------------------------------------------------------------------------
// vertex coords array
static GLfloat s_Points[] = {	1,1,1,  -1,1,1,  -1,-1,1,  1,-1,1,        // v0-v1-v2-v3
								1,1,1,  1,-1,1,  1,-1,-1,  1,1,-1,        // v0-v3-v4-v5
								1,1,1,  1,1,-1,  -1,1,-1,  -1,1,1,        // v0-v5-v6-v1
								-1,1,1,  -1,1,-1,  -1,-1,-1,  -1,-1,1,    // v1-v6-v7-v2
								-1,-1,-1,  1,-1,-1,  1,-1,1,  -1,-1,1,    // v7-v4-v3-v2
								1,-1,-1,  -1,-1,-1,  -1,1,-1,  1,1,-1};   // v4-v7-v6-v5

// normal array
static GLfloat s_Normals[] = {	0,0,1,  0,0,1,  0,0,1,  0,0,1,             // v0-v1-v2-v3
								1,0,0,  1,0,0,  1,0,0, 1,0,0,              // v0-v3-v4-v5
								0,1,0,  0,1,0,  0,1,0, 0,1,0,              // v0-v5-v6-v1
								-1,0,0,  -1,0,0, -1,0,0,  -1,0,0,          // v1-v6-v7-v2
								0,-1,0,  0,-1,0,  0,-1,0,  0,-1,0,         // v7-v4-v3-v2
								0,0,-1,  0,0,-1,  0,0,-1,  0,0,-1};        // v4-v7-v6-v5

// colours array
static GLfloat s_Colours[] = {	1,1,1,  1,1,0,  1,0,0,  1,0,1,              // v0-v1-v2-v3
								1,1,1,  1,0,1,  0,0,1,  0,1,1,              // v0-v3-v4-v5
								1,1,1,  0,1,1,  0,1,0,  1,1,0,              // v0-v5-v6-v1
								1,1,0,  0,1,0,  0,0,0,  1,0,0,              // v1-v6-v7-v2
								0,0,0,  0,0,1,  1,0,1,  1,0,0,              // v7-v4-v3-v2
								0,0,1,  0,0,0,  0,1,0,  0,1,1};             // v4-v7-v6-v5

// uv array
static GLfloat s_UV[] = {		1,1,  1,0,  0,0,  0,1,						// v0-v1-v2-v3
								1,1,  1,0,  0,0,  0,1,						// v0-v3-v4-v5
								1,1,  0,1,  0,0,  1,0,						// v0-v5-v6-v1
								1,1,  0,1,  0,0,  1,0,						// v1-v6-v7-v2
								0,0,  0,1,  1,1,  1,0,						// v7-v4-v3-v2
								0,1,  0,0,  1,0,  1,1};						// v4-v7-v6-v5


//-------------------------------------------------------------------------------------------------
Cube::Cube()
{
	DRD_LOG_DEBUG( L, "constructing a primitive cube" );
}

//-------------------------------------------------------------------------------------------------
Cube::~Cube()
{
	DRD_LOG_DEBUG( L, "destroying a primitive cube" );
}

//-------------------------------------------------------------------------------------------------
void Cube::dumpGL( float lod ) const
{
	DRD_LOG_DEBUG( L, "rendering to GL" );

	// mess up the colours
	for( int i = 0; i < 24; ++i ){
		s_Colours[i*3] = nRand();
		s_Colours[i*3+1] = nRand();
		s_Colours[i*3+2] = nRand();
	}

	// set up arrays
	glEnableClientState( GL_NORMAL_ARRAY);
	glNormalPointer( GL_FLOAT, 0, s_Normals );

	glEnableClientState( GL_COLOR_ARRAY);
	glColorPointer( 3, GL_FLOAT, 0, s_Colours );

	glEnableClientState( GL_VERTEX_ARRAY);
	glVertexPointer( 3, GL_FLOAT, 0, s_Points );

	glEnableClientState( GL_TEXTURE_COORD_ARRAY);
	glTexCoordPointer( 2, GL_FLOAT, 0, s_UV );

	glDrawArrays( GL_QUADS, 0, 24 );

	glDisableClientState( GL_TEXTURE_COORD_ARRAY );
	glDisableClientState( GL_VERTEX_ARRAY );
	glDisableClientState( GL_COLOR_ARRAY );
	glDisableClientState( GL_NORMAL_ARRAY );
}

//-------------------------------------------------------------------------------------------------
void Cube::dumpRib( float lod ) const
{
	DRD_LOG_DEBUG( L, "rendering to Rib" );

	RtInt nverts[] =
	{ 4, 4, 4, 4, 4, 4 };

	RtInt verts[] =
	{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 };

	//	RiBegin("stdout");
	RiSurface("plastic");
	RiPointsPolygons( 6, nverts, verts, "P", s_Points, "N", s_Normals, "Cs", s_Colours, RI_NULL );
	// closing to be done by python
	//	RiArchiveRecord(RI_VERBATIM, "\377");
	//	RiEnd();
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
