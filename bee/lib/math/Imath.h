/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: http://svn/drd/apps/bee/trunk/lib/math/Imath.h $"
 * SVN_META_ID = "$Id: Imath.h 107589 2011-10-13 06:43:12Z stephane.bertout $"
 */

#ifndef bee_math_h
#define bee_math_h
#pragma once

#include "../kernel/types.h"
#include <OpenEXR/ImathMatrix.h>
#include <OpenEXR/ImathVec.h>
#include <OpenEXR/half.h>

//-------------------------------------------------------------------------------------------------
/*
 * new ilmbase-1.0.2 now has Vec4
namespace Imath {
template <class T> class Vec4
{
public:
	Vec4() {}
	Vec4( T a_X, T a_Y, T a_Z, T a_W ) : x( a_X ), y( a_Y ), z( a_Z ), w( a_W ) {}
	Vec4( Imath::Vec3<T> vec3, T w ) : x( vec3.x ), y( vec3.y ), z( vec3.z ), w( w ) {}

	T *			getValue () { return &x; }
	const T *	getValue () const { return &x; }

	T x, y, z, w;
};
} // namespace Imath
*/

namespace bee
{
	// some useful defines
	#define _X_ 0
	#define _Y_ 1
	#define _Z_ 2
	#define _W_ 3

	//-------------------------------------------------------------------------------------------------
	inline float degreesToRadians( float angle )
	{
		return angle * (float) M_PI / 180.0f;
	}
	inline float radiansToDegrees( float angle )
	{
		return angle * 180.0f / (float) M_PI;
	}

	//-------------------------------------------------------------------------------------------------
	template< class T >
	inline T roundUp( T a_Value,
	                  int a_Align )
	{
		return ( a_Value + a_Align - 1 ) & ~( a_Align - 1 );
	}

	//-------------------------------------------------------------------------------------------------
	// Stream output implementation
	template< class T > std::ostream & operator <<( std::ostream &s,
	                                                const Imath::Vec4< T > &v )
	{
		return s << '(' << v.x << ' ' << v.y << ' ' << v.z << ' ' << v.w << ')';
	}

	//! Vec4 is a MATH utility class used to manipulate Vec4 type (x,y,z,w) using Imath implementation
	typedef Imath::Vec4<float> Vec4;
	//! Vec3 is a MATH utility class used to manipulate Vec3 type (x,y,z) using Imath implementation
	typedef Imath::Vec3<float> Vec3;
	//! Vec2 is a MATH utility class used to manipulate Vec2 type (x,y) using Imath implementation
	typedef Imath::Vec2<float> Vec2;
	//! Matrix is a MATH utility class used to manipulate Matrix44 type using Imath implementation
	typedef Imath::Matrix44<float> Matrix;

	typedef Imath::Vec3<half> Vec3h;
	typedef Imath::Vec4<half> Vec4h;

	//! Creates a Perspective Matrix
	void makePerspective( Matrix & a_OutMatrix, float fovy, float aspectRatio, float zNear, float zFar );
	//! Creates an Orthographic Matrix
	void makeOrthographic( Matrix & a_OutMatrix, float orthoWidth, float zNear, float zFar );
	//! Creates a LookAt Matrix
	void makeLookAt( Matrix & a_OutMatrix, const Vec3 & eye, const Vec3 & center, const Vec3 & up );
	//! Printf a Matrix
	void printMatrix( Matrix & a_OutMatrix );

	//! Extract Position from a Matrix
	void getPosition( Vec3 & a_Position, const Matrix & a_Matrix );
	void getPositionTr( Vec3 & a_Position, const Matrix & a_Matrix );
	//! Extract Right vector from a Matrix
	void getRightVector( Vec3 & a_RightVec, const Matrix & a_Matrix );
	void getRightVectorTr( Vec3 & a_RightVec, const Matrix & a_Matrix );
	//! Extract Up vector from a Matrix
	void getUpVector( Vec3 & a_UpVec, const Matrix & a_Matrix );
	void getUpVectorTr( Vec3 & a_UpVec, const Matrix & a_Matrix );
	//! Extract Forward vector from a Matrix
	void getForwardVector( Vec3 & a_ForwardVec, const Matrix & a_Matrix );
	void getForwardVectorTr( Vec3 & a_ForwardVec, const Matrix & a_Matrix );

	//-------------------------------------------------------------------------------------------------
	//! Update Matrix Translation component
	inline void setTranslation ( Matrix & m,
	                             const Vec3 & t )
	{
		m.x[3][0] = t[0];
		m.x[3][1] = t[1];
		m.x[3][2] = t[2];
		m.x[3][3] = 1;
	}

	//-------------------------------------------------------------------------------------------------
	//! Returns Minimum between 2 Vec3
	inline Vec3 VecMin( const Vec3 & a,
	                    const Vec3 & b )
	{
		return Vec3( std::min( a.x, b.x ), std::min( a.y, b.y ), std::min( a.z, b.z ) );
	}

	//-------------------------------------------------------------------------------------------------
	//! Returns Maximum between 2 Vec3
	inline Vec3 VecMax( const Vec3 & a,
	                    const Vec3 & b )
	{
		return Vec3( std::max( a.x, b.x ), std::max( a.y, b.y ), std::max( a.z, b.z ) );
	}

	//-------------------------------------------------------------------------------------------------
	//! Returns if v between a & b
	inline bool VecBetween( const Vec3 & v,
	                    const Vec3 & a,
	                    const Vec3 & b )
	{
		return 	( a.x <= v.x && v.x <= b.x )
			&&	( a.y <= v.y && v.y <= b.y )
			&&	( a.z <= v.z && v.z <= b.z );
	}

	//-------------------------------------------------------------------------------------------------
	//! BBox is a MATH utility class used to create simple Axis Aligned Bounding Box
	class BBox
	{
	public:
		//! Default constructor
		inline BBox() :
			m_Min( Vec3::baseTypeMax() ), m_Max( Vec3::baseTypeMin() )
		{
		}
		//! Constructor
		inline BBox( const Vec3 & a_Min, const Vec3 & a_Max ) :
			m_Min( a_Min ), m_Max( a_Max )
		{
		}

		//! Update the Min and Max using the actual Point parameter
		inline void update( const Vec3 & a_Point )
		{
			if ( m_Min != a_Point && m_Max != a_Point )
			{
				m_Min = VecMin( m_Min, a_Point );
				m_Max = VecMax( m_Max, a_Point );
			}
		}

		//! Returns Min
		inline const Vec3 & getMin() const
		{
			return m_Min;
		}
		//! Returns Max
		inline const Vec3 & getMax() const
		{
			return m_Max;
		}

	private:
		Vec3 m_Min, m_Max;
	};
}

#endif // bee_math_h


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
