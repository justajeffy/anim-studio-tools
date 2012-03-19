
#ifndef ORIENTEDBOUNDINGBOX_H_
#define ORIENTEDBOUNDINGBOX_H_

#include <OpenEXR/ImathVec.h>
#include <OpenEXR/ImathBox.h>
#include <OpenEXR/ImathEuler.h>

template<typename T>
class OrientedBoundingBox
{
public:
	typedef Imath::Vec3<T> Vector;
	typedef Imath::Box<Vector> Box;
	typedef Imath::Matrix44<T> Matrix;
	typedef Imath::Matrix33<T> Matrix3;
	typedef Imath::Euler<T> Euler;

	OrientedBoundingBox()
		: m_Rotation( 0, 0, 0 )
		, m_Box()
	{

	}


	OrientedBoundingBox( const Box & a_Box )
			: m_Rotation( 0, 0, 0 )
			, m_Box( a_Box )
	{

	}

	OrientedBoundingBox( const Euler & a_Rotation, const Box & a_Box )
		: m_Rotation( a_Rotation )
		, m_Box( a_Box )
	{

	}

	bool operator==(const OrientedBoundingBox & a_Other ) const
	{
		return m_Rotation == a_Other.m_Rotation && m_Box == a_Other.m_Box;
	}

	template< typename Iter >
	OrientedBoundingBox( Iter i, Iter e )
	{
		FitBox( i, e );
	}

	template< typename Iter >
	void FitBox( Iter i, Iter e );
	template< typename Iter >
	void FitBoxCovariance( Iter i, Iter e );
	template< typename Iter >
	void FitBoxNoRotation( Iter i, Iter e );

	virtual ~OrientedBoundingBox()
	{

	}

	void ApplyRotation()
	{
		RiRotate( m_Rotation.z*180.0/M_PI, 0, 0, 1 );
		RiRotate( m_Rotation.y*180.0/M_PI, 0, 1, 0 );
		RiRotate( m_Rotation.x*180.0/M_PI, 1, 0, 0 );
	}

	void ApplyReverseRotation()
	{
		RiRotate( -m_Rotation.x*180.0/M_PI, 1, 0, 0 );
		RiRotate( -m_Rotation.y*180.0/M_PI, 0, 1, 0 );
		RiRotate( -m_Rotation.z*180.0/M_PI, 0, 0, 1 );
	}

	template< typename Iter >
	void computeCovarianceMatrix( Iter i, Iter e, Matrix & a_Out);

	const Matrix GetTransform() const
	{
		return m_Rotation.toMatrix44();
	}

	const Matrix GetInverseTransform() const
	{
		return Euler( -m_Rotation.z, -m_Rotation.y, -m_Rotation.x, Euler::ZXY ).toMatrix44();
	}

	const Euler & GetRotation() const
	{
		return m_Rotation;
	}

	const Box & GetBox() const
	{
		return m_Box;
	}

	Box & GetBox()
	{
		return m_Box;
	}

	void makeEmpty()
	{
		m_Box.makeEmpty();
		m_Rotation = Euler();
	}

	template< typename Container >
	void getVerts( Container & a_Vec ) const
	{
		const Vector & min = GetBox().min;
		const Vector & max = GetBox().max;
		Matrix mat = GetTransform();
		for( int i = 0; i < 8; ++i )
		{
			a_Vec.push_back(Vector());
			mat.multVecMatrix( Vector( ( i < 4 ) ? min.x : max.x,
					    	   ( ( i % 4 ) < 2 ) ? min.y : max.y,
								   	   ( i % 2 ) ? min.z : max.z), a_Vec.back() );
		}
	}


private:
	template< typename Iter >
	Box calcTransformedBound( Iter i, Iter e, const Matrix & a_Transform );
	int FindEigenvalues( Matrix & a_Matrix, T o_Evalues[3]);
	void FindEigenvectors( Matrix & a_Matrix, int a_Count, T a_Evalues[3], Vector a_Evectors[3]);


	Euler m_Rotation;
	Box m_Box;
};

#include "OrientedBoundingBoxImpl.h"

#endif /* ORIENTEDBOUNDINGBOX_H_ */


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
