/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: http://svn/drd/apps/bee/trunk/lib/math/Imath.cpp $"
 * SVN_META_ID = "$Id: Imath.cpp 107419 2011-10-12 03:02:53Z stephane.bertout $"
 */

#include "math/Imath.h"
#include <ImathFrustum.h>

using namespace bee;

#define M(row,col)  m[col+row*4]
#define Mi(row,col)  m[col*4+row]

// transposed !
#define SET_ROW(row, v1, v2, v3, v4 )    	\
							M(row,0) = v1; 	\
							M(row,1) = v2; 	\
							M(row,2) = v3; 	\
							M(row,3) = v4

// untransposed !
#define GET_ROW(row, v1, v2, v3, v4 )    	\
							v1 = Mi(row,0); \
							v2 = Mi(row,1); \
							v3 = Mi(row,2); \
							v4 = Mi(row,3)

#define GET_ROW3(row, v1, v2, v3 )    		\
							v1 = Mi(row,0); \
							v2 = Mi(row,1); \
							v3 = Mi(row,2)

#define GET_ROW3_TR(row, v1, v2, v3 )    		\
							v1 = Mi(0,row); \
							v2 = Mi(1,row); \
							v3 = Mi(2,row)

void makeFrustum( float *m,
                  float left,
                  float right,
                  float bottom,
                  float top,
                  float zNear,
                  float zFar )
{
	// note transpose of Matrix_implementation wr.t OpenGL documentation, since the OSG use post multiplication rather than pre.
	float A = ( right + left ) / ( right - left );
	float B = ( top + bottom ) / ( top - bottom );
	float C = -( zFar + zNear ) / ( zFar - zNear );
	float D = -2.0 * zFar * zNear / ( zFar - zNear );
	SET_ROW(0, 2.0*zNear/(right-left), 0.0, 0.0, 0.0 );
	SET_ROW(1, 0.0, 2.0*zNear/(top-bottom), 0.0, 0.0 );
	SET_ROW(2, A, B, C, -1.0 );
	SET_ROW(3, 0.0, 0.0, D, 0.0 )
	;
}

void bee::makePerspective( Matrix & a_OutMatrix,
                           float fovy,
                           float aspectRatio,
                           float zNear,
                           float zFar )
{
	float * m = a_OutMatrix.getValue();

	// calculate the appropriate left, right etc.
	float tan_fovy = tan( degreesToRadians( fovy * 0.5 ) );
	float right = tan_fovy * aspectRatio * zNear;
	float left = -right;
	float top = tan_fovy * zNear;
	float bottom = -top;
	makeFrustum( m, left, right, bottom, top, zNear, zFar );
}

void bee::makeOrthographic( Matrix & a_OutMatrix,
                           float orthoWidth,
                           float zNear,
                           float zFar )
{

	float * m = a_OutMatrix.getValue();
	float ho = orthoWidth * 0.5f;

	Imath::Frustum<float> frustum ( zNear, zFar, -ho, +ho, +ho, -ho, true );
	a_OutMatrix = frustum.projectionMatrix();
}

void set( 	float *m,
			float a00,
			float a01,
			float a02,
			float a03,
			float a10,
			float a11,
			float a12,
			float a13,
			float a20,
			float a21,
			float a22,
			float a23,
			float a30,
			float a31,
			float a32,
			float a33 )
{
	SET_ROW(0, a00, a01, a02, a03 );
	SET_ROW(1, a10, a11, a12, a13 );
	SET_ROW(2, a20, a21, a22, a23 );
	SET_ROW(3, a30, a31, a32, a33 )
	;
}

void bee::printMatrix( Matrix & a_OutMatrix )
{
	float * m = a_OutMatrix.getValue();
	for ( unsigned i = 0 ; i < 4 ; ++i )
	{
		printf("%f %f %f %f \n", M(i,0), M(i,1), M(i,2), M(i,3) );
	}
}

inline void preMultTranslate( float *m,
							  const Vec3 & v )
{
	for ( unsigned i = 0 ; i < 3 ; ++i )
	{
		float tmp = v[ i ];
		if ( tmp == 0 ) continue;

		M(3,0) += tmp * M(i,0);
		M(3,1) += tmp * M(i,1);
		M(3,2) += tmp * M(i,2);
		M(3,3) += tmp * M(i,3);
	}
}

void bee::makeLookAt( Matrix & a_OutMatrix,
                      const Vec3 & eye,
                      const Vec3 & center,
                      const Vec3 & up )
{
	float * m = a_OutMatrix.getValue();

	Vec3 f( center - eye );
	f.normalize();
	Vec3 s( f.cross( up ) );
	s.normalize();
	Vec3 u( s.cross( f ) );
	u.normalize();

	set( m, s[ 0 ], u[ 0 ], -f[ 0 ], 0.0, s[ 1 ], u[ 1 ], -f[ 1 ], 0.0, s[ 2 ], u[ 2 ], -f[ 2 ], 0.0, 0.0, 0.0, 0.0, 1.0 );

	preMultTranslate( m, -eye );
}

void bee::getPosition( Vec3 & a_Position,
                          const Matrix & a_Matrix )
{
	const float * m = a_Matrix.getValue();
	GET_ROW3( _W_, a_Position.x, a_Position.y, a_Position.z )
	;
}

void bee::getPositionTr( Vec3 & a_Position,
                          const Matrix & a_Matrix )
{
	const float * m = a_Matrix.getValue();
	GET_ROW3_TR( _W_, a_Position.x, a_Position.y, a_Position.z )
	;
}

void bee::getRightVector( Vec3 & a_RightVec,
                          const Matrix & a_Matrix )
{
	const float * m = a_Matrix.getValue();
	GET_ROW3( _X_, a_RightVec.x, a_RightVec.y, a_RightVec.z )
	;
}

void bee::getRightVectorTr( Vec3 & a_RightVec,
                          const Matrix & a_Matrix )
{
	const float * m = a_Matrix.getValue();
	GET_ROW3_TR( _X_, a_RightVec.x, a_RightVec.y, a_RightVec.z )
	;
}

void bee::getUpVector( Vec3 & a_UpVec,
                       const Matrix & a_Matrix )
{
	const float * m = a_Matrix.getValue();
	GET_ROW3( _Y_, a_UpVec.x, a_UpVec.y, a_UpVec.z )
	;
}

void bee::getUpVectorTr( Vec3 & a_UpVec,
                       const Matrix & a_Matrix )
{
	const float * m = a_Matrix.getValue();
	GET_ROW3_TR( _Y_, a_UpVec.x, a_UpVec.y, a_UpVec.z )
	;
}

void bee::getForwardVector( Vec3 & a_ForwardVec,
                            const Matrix & a_Matrix )
{
	const float * m = a_Matrix.getValue();
	GET_ROW3( _Z_, a_ForwardVec.x, a_ForwardVec.y, a_ForwardVec.z )
	;
}

void bee::getForwardVectorTr( Vec3 & a_ForwardVec,
                              const Matrix & a_Matrix )
{
	const float * m = a_Matrix.getValue();
	GET_ROW3_TR( _Z_, a_ForwardVec.x, a_ForwardVec.y, a_ForwardVec.z )
	;
}

#undef M
#undef SET_ROW
#undef GET_ROW


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
