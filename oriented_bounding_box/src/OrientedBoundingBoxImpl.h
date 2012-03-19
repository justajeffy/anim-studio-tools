
#ifndef OBB_IMPL_H
#define OBB_IMPL_H

#include "OrientedBoundingBox.h"
#include <OpenEXR/ImathMatrix.h>


//-------------------------------------------------------------------------------------------------
template< typename T >
template< typename Iter >
Imath::Box<Imath::Vec3<T> > OrientedBoundingBox<T>::calcTransformedBound( Iter i, Iter e, const Matrix & a_Transform )
{
	Box result;

	for ( ; i != e ; ++i )
	{
		Vector out;
		a_Transform.multVecMatrix( *i, out );
		result.extendBy( out );
	}

	return result;
}

//-------------------------------------------------------------------------------------------------
template< typename T >
template< typename Iter >
void OrientedBoundingBox<T>::FitBox( Iter i, Iter e )
{
	m_Rotation = Euler( 0, 0, 0);
	if ( i == e )
	{
		m_Box = Box();
		return;
	}

	m_Box = calcTransformedBound( i, e, Matrix() );
	Imath::V3f size = m_Box.size();
	float volume = size.x*size.y*size.z;
	float baseVolume = volume;

	static int NUM_STEPS_X = 4;
	static int NUM_STEPS_Y = 8;
	static int NUM_STEPS_Z = 8;
	for ( int x = 0; x < NUM_STEPS_X; ++x )
	{
		float xv = (x * M_PI/2 )/(NUM_STEPS_X);
		for ( int y = 0; y < NUM_STEPS_Y; ++y )
		{
			float yv = ( y * M_PI ) / (NUM_STEPS_Y);
			for( int z = 0; z < NUM_STEPS_Z; ++z )
			{
				float zv = ( z * M_PI ) / (NUM_STEPS_Z);
				Euler e2( -zv, -yv, -xv, Imath::Euler<T>::ZYX );
				Matrix reverse( e2.toMatrix44() );
				Box result = calcTransformedBound( i, e, reverse );
				size = result.size();
				if( size.x*size.y*size.z < volume )
				{
					volume = size.x*size.y*size.z;
					m_Box = result;
					m_Rotation = Imath::Eulerf( xv, yv, zv );
				}
			}
		}
	}
}

//-------------------------------------------------------------------------------------------------
template< typename T >
template< typename Iter >
void OrientedBoundingBox<T>::FitBoxNoRotation( Iter i, Iter e )
{
	m_Rotation = Euler( 0, 0, 0 );
	if ( i == e )
	{
		m_Box = Box();
		return;
	}
	m_Box = calcTransformedBound( i, e, Matrix() );
}

#define COPY_ROW( to, from ) to[0] = from[0]; to[1] = from[1]; to[2] = from[2];
#define ADD_ROW( to, from, factor ) to[0] += from[0]*factor;\
		to[1] += from[1]*factor; to[2] += from[2]*factor;

template<typename T>
T cubeRoot( T a_In )
{
	if( a_In < 0 )
	{
		return -pow( -a_In, 1.f/3.f );
	}
	else
	{
		return pow( a_In, 1.f/3.f );
	}
}

template<typename T>
T clamp( T in, T a_Min, T a_Max )
{
	return ( in < a_Min ) ? a_Min : ( ( in > a_Max ) ? a_Max : in );
}

//#define EIGENVALUE_DEBUG

//-------------------------------------------------------------------------------------------------
template< typename T >
int OrientedBoundingBox<T>::FindEigenvalues( Matrix & a_Matrix, T o_Evalues[3])
{
	T a /*x^3*/= -1;
	T b /*x^2*/ = a_Matrix[0][0] + a_Matrix[1][1] + a_Matrix[2][2];
	T c /*x*/ = -( a_Matrix[0][0] * a_Matrix[1][1] )
					   -( a_Matrix[0][0] * a_Matrix[2][2] )
					   -( a_Matrix[1][1] * a_Matrix[2][2] )
					   +( a_Matrix[0][1] * a_Matrix[1][0] )
					   +( a_Matrix[0][2] * a_Matrix[2][0] )
					   +( a_Matrix[2][1] * a_Matrix[1][2] );
	T d /*1*/ = a_Matrix[0][0] * ( a_Matrix[1][1] * a_Matrix[2][2] - a_Matrix[2][1] * a_Matrix[1][2] )
						-a_Matrix[1][0] * ( a_Matrix[0][1] * a_Matrix[2][2] - a_Matrix[2][1] * a_Matrix[0][2] )
						-a_Matrix[2][0] * ( a_Matrix[1][1] * a_Matrix[0][2] - a_Matrix[0][1] * a_Matrix[1][2] );
	T p = (3*a*c-b*b)/(3*a*a);
	T q = (2*b*b*b-9*a*b*c+27*a*a*d)/(27*a*a*a);
#ifdef EIGENVALUE_DEBUG
	printf("%fx^3+%fx^2+%fx+%f\n",a,b,c,d);
#endif
	if( p > 0 )
	{
		T term = 1.5*q/p*sqrt(3./p);
		o_Evalues[0] = -2*sqrt(p/3)*sinh(0.33333*asinh(term)) - b / ( 3 * a );
#ifdef EIGENVALUE_DEBUG
		printf("p = %f, q = %f, term = %f, positive hyperbolic\n", p, q, term);
		printf("roots: %f only\n", o_Evalues[0]);
	#define EQN(X) a*X*X*X+b*X*X+c*X+d
		printf("%f\n", EQN(o_Evalues[0]) );
	#undef EQN
#endif
		return 1;
	}
	else if( 4*p*p*p+27*q*q > 0 )
	{
		T sign = ( q > 0 ) ? 1 : -1;
		T term = -1.5*fabs(q)/p*sqrt(-3./p);
		o_Evalues[0] = -2*sign*sqrt(-p/3)*cosh(0.33333*acosh(term)) - b / ( 3 * a );
#ifdef EIGENVALUE_DEBUG
		printf("p = %f, q = %f, term = %f, hyperbolic\n", p, q, term);
		printf("roots: %f only\n", o_Evalues[0]);
	#define EQN(X) a*X*X*X+b*X*X+c*X+d
		printf("%f\n", EQN(o_Evalues[0]) );
	#undef EQN
#endif
		return 1;
	}
	else
	{
		T term = clamp(1.5*q/p*sqrt(-3./p), -1., 1.);
		for( int k = 0; k < 3; ++k )
		{
			o_Evalues[k] = 2*sqrt(-p/3)*cos(0.33333*acos(term)-k*2*M_PI/3) - b / ( 3 * a );
		}
#ifdef EIGENVALUE_DEBUG
		printf("p = %f, q = %f, term = %f, cosine\n", p, q, term);
		printf("roots: %f, %f, %f\n", o_Evalues[0], o_Evalues[1], o_Evalues[2]);
	#define EQN(X) a*X*X*X+b*X*X+c*X+d
		printf("%f; %f; %f\n", EQN(o_Evalues[0]), EQN(o_Evalues[1]), EQN(o_Evalues[2]));
	#undef EQN
#endif
		return 3;
	}
}

template<typename T>
bool NearZero( T data[3] )
{
	return data[0] > -0.0001f && data[0] < 0.0001f &&
		   data[1] > -0.0001f && data[1] < 0.0001f &&
		   data[2] > -0.0001f && data[2] < 0.0001f;
}

//-------------------------------------------------------------------------------------------------
template< typename T >
void OrientedBoundingBox<T>::FindEigenvectors( Matrix & a_Matrix, int a_Count, T a_Evalues[3], Vector a_Evectors[3])
{
	for( int i = 0; i < a_Count; ++i )
	{
		Matrix mat( a_Matrix );
		mat[0][0] -= a_Evalues[i];
		mat[1][1] -= a_Evalues[i];
		mat[2][2] -= a_Evalues[i];
		if( mat[0][0] == 0 )
		{
			Vector temp( mat[0][0], mat[0][1], mat[0][2] );
			COPY_ROW( mat[0], mat[2] );
			COPY_ROW( mat[2], temp );
		}
		else
		{
			T factor = mat[2][0] / mat[0][0];
			ADD_ROW( mat[2], mat[0], -factor );
		}

		if( mat[0][0] == 0 )
		{
			Vector temp( mat[0][0], mat[0][1], mat[0][2] );
			COPY_ROW( mat[0], mat[1] );
			COPY_ROW( mat[1], temp );
		}
		else
		{
			T factor = mat[1][0] / mat[0][0];
			ADD_ROW( mat[1], mat[0], -factor );
		}

		if( mat[1][1] == 0 )
		{
			Vector temp( mat[1][0], mat[1][1], mat[1][2] );
			COPY_ROW( mat[1], mat[2] );
			COPY_ROW( mat[2], temp );
		}
		else
		{
			T factor = mat[2][1] / mat[1][1];
			ADD_ROW( mat[2], mat[1], -factor );
		}

		// Matrix is now of the form:
		// a b c
		// 0 d e
		// 0 0 0

		/*
		if( !NearZero( mat[2] ) )
		{
			// Note that this is sometimes possible through numerical error.
			DRD_LOG_DEBUG("non-singular eignenmatrix? evalue is %f, matrix is: \n(%f,%f,%f)\n(%f,%f,%f)\n(%f,%f,%f)\n",
					a_Evalues[i],
					mat[0][0], mat[0][1], mat[0][2],
					mat[1][0], mat[1][1], mat[1][2],
					mat[2][0], mat[2][1], mat[2][2]);
		}
		*/


		if( NearZero( mat[0] ) )
		{
#ifdef EIGENVALUE_DEBUG
		printf("Zero matrix; all using identity evectors");
#endif
			// Matrix is a multiple of the identity matrix; all eigenvalues the same
			a_Evectors[0] = Vector(1.f,0.f,0.f);
			a_Evectors[1] = Vector(0.f,1.f,0.f);
			a_Evectors[2] = Vector(0.f,0.f,1.f);
			return;
		}
		else if( NearZero( mat[1] ) )
		{
			// Matrix has two zero rows; two identical eigenvalues
			int last = a_Count-1;
			if( i == 0 )
			{
				if( a_Evalues[1] == a_Evalues[0] )
				{
					a_Evalues[1] = a_Evalues[2];
					a_Evalues[2] = a_Evalues[0];
				}
			}
			--a_Count;
			Vector first( mat[0][0], mat[0][1], mat[0][2] );
			if( first[2] < 0.8 && first[2] > -0.8 )
			{
				a_Evectors[i] = first.cross( Vector(0,0,1) );
			}
			else
			{
				a_Evectors[i] = first.cross( Vector(0,1,0) );
			}
			a_Evectors[i].normalize();
			a_Evectors[last] = first.cross(a_Evectors[i]);
			a_Evectors[last].normalize();
		}
		else
		{
			a_Evectors[i] = Vector(mat[0][0], mat[0][1], mat[0][2]).
					cross( Vector(mat[1][0], mat[1][1], mat[1][2]) );
			a_Evectors[i].normalize();
		}
	}
#undef COPY_ROW
#undef ADD_ROW
}

//-------------------------------------------------------------------------------------------------
template< typename T >
template< typename Iter >
void OrientedBoundingBox<T>::FitBoxCovariance( Iter i, Iter e )
{
	Matrix covarianceMatrix;
	computeCovarianceMatrix( i, e, covarianceMatrix );

	T evalues[3];
	int numEvalues = FindEigenvalues( covarianceMatrix, evalues );
	Vector evectors[3];
	FindEigenvectors( covarianceMatrix, numEvalues, evalues, evectors );
	// Construct a basis from the eigenvalues
	if( numEvalues == 1 )
	{
		if( evectors[0][0] < 0.8 && evectors[0][0] > -0.8 )
		{
			evectors[1] = evectors[0].cross(Imath::V3f(1,0,0));
		}
		else
		{
			evectors[1] = evectors[0].cross(Imath::V3f(0,1,0));
		}
	}
	else
	{
		evectors[1] = evectors[0].cross(evectors[1]);
	}
	evectors[2] = evectors[0].cross(evectors[1]);
	Matrix3 transformMatrix(evectors[0][0], evectors[0][1], evectors[0][2],
	                            evectors[1][0], evectors[1][1], evectors[1][2],
	                            evectors[2][0], evectors[2][1], evectors[2][2]);
	m_Rotation = Euler( transformMatrix );
	Euler rot1 = Euler( -m_Rotation.z, -m_Rotation.y, -m_Rotation.x, Euler::ZYX );
	m_Box = calcTransformedBound( i, e, rot1.toMatrix44() );
}

//-------------------------------------------------------------------------------------------------
template< typename T >
template< typename Iter>
void OrientedBoundingBox<T>::computeCovarianceMatrix( Iter i, Iter e,
                             			  Matrix & a_Out)
{
	double means[3] = {0., 0., 0.};
	int count = 0;
	for ( Iter it = i; it != e; ++it)
	{
		means[0] += it->x;
		means[1] += it->y;
		means[2] += it->z;
		++count;
	}
	means[0] /= count;
	means[1] /= count;
	means[2] /= count;

	a_Out = Matrix( 0.f );
	a_Out[3][3] = 1;

	for (int x = 0; x < 3; ++x )
	{
		for( int y = 0; y < 3; ++y )
		{
			for (Iter it = i; it != e; ++it)
			{
				a_Out[x][y] += (means[x] - (*it)[x]) *
							   (means[y] - (*it)[y]);
			}
			a_Out[x][y] /= count;
		}
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
