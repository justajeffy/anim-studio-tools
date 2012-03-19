/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: $"
 * SVN_META_ID = "$Id: device_vector_algorithms.h 71751 2011-03-14 01:49:21Z chris.cooper $"
 */

#ifndef grind_device_vector_algorithms_h
#define grind_device_vector_algorithms_h

//-------------------------------------------------------------------------------------------------
#include "bbox.h"
#include "device_vector.h"

//! the grind namespace
namespace grind {

	//-------------------------------------------------------------------------------------------------
	//! perturb a float array with the user-defined deviation
	void perturb(	unsigned int i_Seed, //! random number seed
					float i_Deviation, //! magnitude of deviation
					float i_Gamma, //! bias the distribution
					DeviceVectorHandle< float >& o_Result //! vector to be modified
			);

	//-------------------------------------------------------------------------------------------------
	//! perturb a float array with the user-defined deviation
	void reproducablePerturb(	unsigned int i_Seed, //! random number seed
	                         	float i_Deviation, //! magnitude of deviation
	                         	float i_Gamma, //! bias the distribution
	                         	const DeviceVector< int >& i_Id, //! ids (same size as result )
	                         	DeviceVector< float >& o_Result //! vector to be modified
							);

	//-------------------------------------------------------------------------------------------------
	void reproducableRandomSample(	int i_Seed, //! random number seed
	                              	int i_Stride, //! stride of increments through random number table
	                              	const DeviceVector< int >& i_Id, //! ids (same size as result )
	                              	DeviceVector< float >& o_Result //! vector to be modified
									);

	//-------------------------------------------------------------------------------------------------
	//! remap a float array
	void remap( float i_Min, //! zero will be mapped to this value
				float i_Max, //! one will be mapped to this value
				DeviceVectorHandle< float >& o_Result //! vector to be modified
			);

	//-------------------------------------------------------------------------------------------------
	//! set all elements to a value
	template< typename T >
	void setAllElements( const T& i_Val, DeviceVector<T>& o_Dst );

	//-------------------------------------------------------------------------------------------------
	//! return the max element value
	template< typename T >
	T getMaxValue( const DeviceVector<T>& i_Src );

	//-------------------------------------------------------------------------------------------------
	//! return the min element value
	template< typename T >
	T getMinValue( const DeviceVector<T>& i_Src );

	//-------------------------------------------------------------------------------------------------
	//! inclusive scan of a vector (http://www.sgi.com/tech/stl/partial_sum.html)
	template< typename T >
	void inclusiveScan( const DeviceVector<T>& i_Src, DeviceVector<T>& o_Dst );

	//-------------------------------------------------------------------------------------------------
	//! adjacent difference, or opposite of a scan
	//! (http://www.sgi.com/tech/stl/adjacent_difference.html)
	template< typename T >
	void adjacentDifference( const DeviceVector<T>& i_Src, DeviceVector<T>& o_Dst );

	//-------------------------------------------------------------------------------------------------
	//! parallel reduction of a vector (sum of all elements)
	template< typename T >
	T reduce( const DeviceVector<T>& i_Src );

	//! @cond DEV

	//-------------------------------------------------------------------------------------------------
	//! in-place compact of a device vector
	template< typename T >
	void compact(	const DeviceVector<bool>& i_Exists, DeviceVector<T>& o_Result );

	//-------------------------------------------------------------------------------------------------
	//! run length encoding
	/*! \note: all vectors will be of length i_Src(), actual elements returned by function
	 */
	template< typename T >
	int runLengthEncode( const DeviceVector<T>& i_Src, DeviceVector<T>& o_Dst, DeviceVector<int>& o_Lengths );

	//-------------------------------------------------------------------------------------------------
	//! swap internal data
	template< typename T >
	void swap( DeviceVector<T>& o_A, DeviceVector<T>& o_B );

	//-------------------------------------------------------------------------------------------------
	//! bounding box calculation
	BBox getBounds( const DeviceVector<Imath::V3f>& i_Src );

	//-------------------------------------------------------------------------------------------------
	//! bounding box calculation from an indexed set of points
	BBox getBounds( unsigned int i_Count, int* i_Indices, const DeviceVector<Imath::V3f>& i_Src );

	//-------------------------------------------------------------------------------------------------
	//! copy from an int vec to a float vec
	void copy( const DeviceVector<int>& i_Src, DeviceVector<float>& o_Dst );

	//-------------------------------------------------------------------------------------------------
	template< typename T>
	void copy( const DeviceVector<T>& i_Src, HostVector<T>& o_Dst )
	{
		assert( o_Dst.size() == i_Src.size() );
		i_Src.getValue( o_Dst );
	}

	template< typename T1, typename T2 >
	void copy( const DeviceVector<T1>& i_Src, HostVector<T2>& o_Dst )
	{
		assert( o_Dst.size() == i_Src.size() );

		HostVector<T1> temp;
		i_Src.getValue( temp );
		std::copy( temp.begin(), temp.end(), o_Dst.begin() );
	}

	//-------------------------------------------------------------------------------------------------
	template< typename T >
	void save( const DeviceVector<T>& i_src, const std::string& i_path );

	//! @endcond

}

#endif /* grind_device_vector_algorithms_h */


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
