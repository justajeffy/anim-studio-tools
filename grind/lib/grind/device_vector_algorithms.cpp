/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: $"
 * SVN_META_ID = "$Id: device_vector_algorithms.cpp 99443 2011-08-30 01:14:50Z hugh.rayner $"
 */

//-------------------------------------------------------------------------------------------------
#include <drdDebug/log.h>
DRD_MKLOGGER(L,"drd.grind.DeviceVectorAlgorithms");

#include "device_vector.h"
#include "host_vector.h"
#include "device_vector_algorithms.h"
#include "random.h"
#include "bbox.h"

#include <algorithm>

#include <cuda_runtime_api.h>
#include <fstream>


//-------------------------------------------------------------------------------------------------
using namespace grind;
using namespace drd;

//-------------------------------------------------------------------------------------------------
extern "C"
void
perturbDevice( 	unsigned int i_Seed,
				float i_Devaiation,
				float i_Gamma,
				unsigned int i_TableSize,
				const float* i_Table,
				unsigned int i_ResultSize,
				float* o_Result );

//-------------------------------------------------------------------------------------------------
extern "C"
void
reproducablePerturbDevice( 	unsigned int i_Seed,
							float i_Devaiation,
							float i_Gamma,
							unsigned int i_TableSize,
							const float* i_Table,
							unsigned int i_ResultSize,
							const int* i_Id,
							float* o_Result );



extern "C"
void
remapDevice(	float i_Min,
            	float i_Max,
            	unsigned int i_ResultSize,
            	float* o_Result );

extern "C"
void
getBoundsDevice(	unsigned int i_Count,
                	const Imath::V3f* i_Data,
                	Imath::V3f& o_ResultMin, Imath::V3f& o_ResultMax );

extern "C"
void
getIndexedBoundsDevice(	unsigned int i_Count,
                       	int* i_Indices,
                       	const Imath::V3f* i_Data,
                       	Imath::V3f& o_ResultMin, Imath::V3f& o_ResultMax );

extern "C"
void
compactFloatDevice( size_t i_Count, bool* i_Exists, size_t& o_ResultCount, float* o_Result );

extern "C"
void
compactIntDevice( size_t i_Count, bool* i_Exists, size_t& o_ResultCount, int* o_Result );

extern "C"
void
copyIntToFloatDevice( size_t i_Count, int* i_Src, float* o_Result );

extern "C"
void
reproducableRandomSampleDevice( 	int i_Seed,
                                	int i_Stride,
                                	int i_RandomTableSize,
                                	const float* i_RandomTable,
                                	unsigned int i_ResultSize,
                                	const int* i_Id,
                                	float* o_Result );

namespace grind {

template< typename T >
void setAllElementsDevice( const T& i_Val, unsigned int i_ResultSize, T* o_Result );

template< typename T >
void getMaxValueDevice( size_t i_Count, T* i_Src, T& o_Result );

template< typename T >
void getMinValueDevice( size_t i_Count, T* i_Src, T& o_Result );

template< typename T >
void inclusiveScanDevice( size_t i_Count, const T* i_Src, T* o_Result );

template< typename T >
void adjacentDifferenceDevice( size_t i_Count, const T* i_Src, T* o_Result );

template< typename T >
int runLengthEncodeDevice( size_t i_Count, const T* i_Src, T* o_Result, int* o_Lengths );

template< typename T >
T reduceDevice( size_t i_Count, const T* i_Src );

//-------------------------------------------------------------------------------------------------
void perturb( 	unsigned int i_Seed,
						float i_Deviation,
						float i_Gamma,
						DeviceVectorHandle< float >& o_Result )
{
	// access the random number table
	const DeviceVector< float >& rng = DeviceRandom::instance().getNormRandTable();

	// now run the perturb algorithm on the gpu
	perturbDevice( i_Seed, i_Deviation, i_Gamma, rng.size(), rng.getDevicePtr(), o_Result.get().size(), o_Result.get().getDevicePtr() );
}

//-------------------------------------------------------------------------------------------------
void reproducablePerturb( 	unsigned int i_Seed,
                          	float i_Deviation,
                          	float i_Gamma,
                          	const DeviceVector<int>& i_Id,
                          	DeviceVector< float >& o_Result )
{
	if( i_Id.size() != o_Result.size() )
		throw std::runtime_error( "ids and result must be the same size for reproducablePerturb" );

	// access the random number table
	const DeviceVector< float >& rng = DeviceRandom::instance().getNormRandTable();

	// now run the perturb algorithm on the gpu
	reproducablePerturbDevice( i_Seed, i_Deviation, i_Gamma, rng.size(), rng.getDevicePtr(), o_Result.size(), i_Id.getDevicePtr(), o_Result.getDevicePtr() );
}

//-------------------------------------------------------------------------------------------------
void reproducableRandomSample( 	int i_Seed,
                               	int i_Stride,
                               	const DeviceVector<int>& i_Id,
                               	DeviceVector< float >& o_Result )
{
	if( i_Id.size() != o_Result.size() )
		throw std::runtime_error( "ids and result must be the same size for reproducablePerturb" );

	// access the random number table
	const DeviceVector< float >& rng = DeviceRandom::instance().getNormRandTable();

	// now run the perturb algorithm on the gpu
	reproducableRandomSampleDevice( i_Seed, i_Stride, rng.size(), rng.getDevicePtr(), o_Result.size(), i_Id.getDevicePtr(), o_Result.getDevicePtr() );
}

//-------------------------------------------------------------------------------------------------
void remap( float i_Min, float i_Max, DeviceVectorHandle< float >& o_Result )
{
	// now run the perturb algorithm on the gpu
	remapDevice( i_Min, i_Max, o_Result.get().size(), o_Result.get().getDevicePtr() );
}

//-------------------------------------------------------------------------------------------------
template< typename T >
void setAllElements( 	const T& i_Val,
						DeviceVector< T >& o_Dst )
{
	setAllElementsDevice( i_Val, o_Dst.size(), o_Dst.getDevicePtr() );
}

// explicit instantiation
template void setAllElements<int>( const int& i_Val, DeviceVector<int>& o_Dst );
template void setAllElements<float>( const float& i_Val, DeviceVector<float>& o_Dst );
template void setAllElements<Imath::V2f>( const Imath::V2f& i_Val, DeviceVector<Imath::V2f>& o_Dst );
template void setAllElements<Imath::V3f>( const Imath::V3f& i_Val, DeviceVector<Imath::V3f>& o_Dst );
template void setAllElements<Imath::V4f>( const Imath::V4f& i_Val, DeviceVector<Imath::V4f>& o_Dst );

//-------------------------------------------------------------------------------------------------
template< typename T >
T getMaxValue( const DeviceVector<T>& i_Src )
{
	T mv;
	getMaxValueDevice( i_Src.size(), i_Src.getDevicePtr(), mv );
	return mv;
}

// explicit instantiation
template int getMaxValue<int>( const DeviceVector<int>& i_Src );
template float getMaxValue<float>( const DeviceVector<float>& i_Src );

//-------------------------------------------------------------------------------------------------
template< typename T >
T getMinValue( const DeviceVector<T>& i_Src )
{
	T mv;
	getMinValueDevice( i_Src.size(), i_Src.getDevicePtr(), mv );
	return mv;
}

// explicit instantiation
template int getMinValue<int>( const DeviceVector<int>& i_Src );
template float getMinValue<float>( const DeviceVector<float>& i_Src );

//-------------------------------------------------------------------------------------------------
template< typename T >
void inclusiveScan( const DeviceVector<T>& i_Src, DeviceVector<T>& o_Dst )
{
	assert( i_Src.size() == o_Dst.size() );
	inclusiveScanDevice( i_Src.size(), i_Src.getDevicePtr(), o_Dst.getDevicePtr() );
}

// explicit instantiation
template void inclusiveScan<int>( const DeviceVector<int>& i_Src, DeviceVector<int>& o_Dst );
template void inclusiveScan<float>( const DeviceVector<float>& i_Src, DeviceVector<float>& o_Dst );

//-------------------------------------------------------------------------------------------------
template< typename T >
void adjacentDifference( const DeviceVector<T>& i_Src, DeviceVector<T>& o_Dst )
{
	assert( i_Src.size() == o_Dst.size() );
	adjacentDifferenceDevice( i_Src.size(), i_Src.getDevicePtr(), o_Dst.getDevicePtr() );
}

// explicit instantiation
template void adjacentDifference<int>( const DeviceVector<int>& i_Src, DeviceVector<int>& o_Dst );
template void adjacentDifference<float>( const DeviceVector<float>& i_Src, DeviceVector<float>& o_Dst );


//-------------------------------------------------------------------------------------------------
template< typename T >
int runLengthEncode( const DeviceVector<T>& i_Src, DeviceVector<T>& o_Dst, DeviceVector<int>& o_Lengths )
{
	if( i_Src.size() == 0 ) return 0;
	assert( o_Dst.size() == i_Src.size() );
	assert( o_Lengths.size() == i_Src.size() );

	return runLengthEncodeDevice( i_Src.size(), i_Src.getDevicePtr(), o_Dst.getDevicePtr(), o_Lengths.getDevicePtr() );
}

// explicit instantiations
template int runLengthEncode<int>( const DeviceVector<int>& i_Src, DeviceVector<int>& o_Dst, DeviceVector<int>& o_Lengths );


//-------------------------------------------------------------------------------------------------
template< typename T >
T reduce( const DeviceVector<T>& i_Src )
{
	return reduceDevice( i_Src.size(), i_Src.getDevicePtr() );
}

// explicit instantiation
template int reduce<int>( const DeviceVector<int>& i_Src );
template float reduce<float>( const DeviceVector<float>& i_Src );


//-------------------------------------------------------------------------------------------------
//TODO: combine these implementations
template<>
void compact<float>(	const DeviceVector<bool>& i_Exists, DeviceVector<float>& o_Result )
{
	// early exit
	if( i_Exists.size() == 0 && o_Result.size() == 0 ) return;

	DRD_LOG_ASSERT( L, i_Exists.isOk(), "The 'exists' deviceVector is not ok." );
	DRD_LOG_ASSERT( L, o_Result.isOk(), "The 'results' deviceVector is not ok." );
	DRD_LOG_ASSERT( L, i_Exists.size() == o_Result.size(), "exists and result array sizes don't match: " << i_Exists.size() << "!=" << o_Result.size() );

	size_t n_elements = i_Exists.size();
	size_t n_valid = 0;

	compactFloatDevice( n_elements, i_Exists.getDevicePtr(), n_valid, o_Result.getDevicePtr() );

	// TODO: this may be suboptimal.  might want to support the concept of std::copy()

	// copy to temp
	DeviceVector<float> temp;
	temp.resize( n_valid );
#ifdef __DEVICE_EMULATION__
	memcpy( temp.getDevicePtr(), o_Result.getDevicePtr(), n_valid * sizeof(float) );
#else
	SAFE_CUDA( cudaMemcpy( temp.getDevicePtr(), o_Result.getDevicePtr(), n_valid * sizeof(float), cudaMemcpyDeviceToDevice ) );
#endif

	// copy back to result
	o_Result.resize( n_valid );
#ifdef __DEVICE_EMULATION__
	memcpy( o_Result.getDevicePtr(), temp.getDevicePtr(), n_valid * sizeof(float) );
#else
	SAFE_CUDA( cudaMemcpy( o_Result.getDevicePtr(), temp.getDevicePtr(), n_valid * sizeof(float), cudaMemcpyDeviceToDevice ) );
#endif
}

//-------------------------------------------------------------------------------------------------
//TODO: combine these implementations
template<>
void compact<int>(	const DeviceVector<bool>& i_Exists, DeviceVector<int>& o_Result )
{
	// early exit
	if( i_Exists.size() == 0 && o_Result.size() == 0 ) return;

	DRD_LOG_ASSERT( L, i_Exists.isOk(), "The 'exists' deviceVector is not ok." );
	DRD_LOG_ASSERT( L, o_Result.isOk(), "The 'results' deviceVector is not ok." );
	DRD_LOG_ASSERT( L, i_Exists.size() == o_Result.size(), "exists and result array sizes don't match: " << i_Exists.size() << "!=" << o_Result.size() );

	size_t n_elements = i_Exists.size();
	size_t n_valid = 0;

	compactIntDevice( n_elements, i_Exists.getDevicePtr(), n_valid, o_Result.getDevicePtr() );

	// TODO: this may be suboptimal but should be robust (gl interop etc).  might want to support the concept of std::swap()

	// copy to temp
	DeviceVector<int> temp;
	temp.resize( n_valid );
#ifdef __DEVICE_EMULATION__
	memcpy( temp.getDevicePtr(), o_Result.getDevicePtr(), n_valid * sizeof(int) );
#else
	SAFE_CUDA( cudaMemcpy( temp.getDevicePtr(), o_Result.getDevicePtr(), n_valid * sizeof(int), cudaMemcpyDeviceToDevice ) );
#endif

	// copy back to result
	o_Result.resize( n_valid );
#ifdef __DEVICE_EMULATION__
	memcpy(  o_Result.getDevicePtr(), temp.getDevicePtr(), n_valid * sizeof(int) );
#else
	SAFE_CUDA( cudaMemcpy( o_Result.getDevicePtr(), temp.getDevicePtr(), n_valid * sizeof(int), cudaMemcpyDeviceToDevice ) );
#endif
}

//-------------------------------------------------------------------------------------------------
template< typename T >
void swap( DeviceVector<T>& i_A, DeviceVector<T>& i_B )
{
	LOGLINE();
	throw std::runtime_error( "swap not supported yet\n" );
}

//-------------------------------------------------------------------------------------------------
void copy( const DeviceVector<int>& i_Src, DeviceVector<float>& o_Dst )
{
	assert( i_Src.size() == o_Dst.size() );
	copyIntToFloatDevice( i_Src.size(), i_Src.getDevicePtr(), o_Dst.getDevicePtr() );
}

//-------------------------------------------------------------------------------------------------
BBox getBoundsGold( const DeviceVector<Imath::V3f>& i_Src )
{
	if( i_Src.size() == 0 ) return BBox();

	HostVector<Imath::V3f> h_Src;
	i_Src.getValue( h_Src );
	BBox bounds;
	bounds.populate( h_Src.begin(), h_Src.end(), 0 );
	return bounds;
}

//-------------------------------------------------------------------------------------------------
BBox getBounds( const DeviceVector<Imath::V3f>& i_Src )
{
	if( i_Src.size() == 0 ) return BBox();

	BBox result;
	getBoundsDevice( i_Src.size(), i_Src.getDevicePtr(), result.GetBox().min, result.GetBox().max );

	return result;
}

//-------------------------------------------------------------------------------------------------
BBox getBounds( unsigned int i_Count, int* i_Indices, const DeviceVector<Imath::V3f>& i_Src )
{
	if( i_Count == 0 ) return BBox();

	BBox result;
	getIndexedBoundsDevice( i_Count, i_Indices, i_Src.getDevicePtr(), result.GetBox().min, result.GetBox().max );

	return result;
}

float fRand() { return rand()/((double)RAND_MAX + 1); }
Imath::V3f genRandVec(){ return Imath::V3f(fRand(),fRand(),fRand()) * 10 - Imath::V3f(5,5,5); }

//-------------------------------------------------------------------------------------------------
// unit test for bound calculation
bool testGetBounds()
{
	DRD_LOG_INFO( L, "testGetBounds()" );

	DeviceVector<Imath::V3f> d_data;
	HostVector<Imath::V3f> h_data;

	h_data.resize( 1024 );
	std::generate( h_data.begin(), h_data.end(), genRandVec );
	d_data.setValue( h_data );

	BBox bounds_gold = getBoundsGold( d_data );
	BBox bounds = getBounds( d_data );

	// make sure the results are both equal and correct
	bool result = ( bounds == bounds_gold )
			&& ( (bounds.GetBox().min - Imath::V3f(-5,-5,-5)).length() < 0.1)
			&& ( (bounds.GetBox().max - Imath::V3f( 5, 5, 5)).length() < 0.1);

	DRD_LOG_INFO( L, "bounds min: " << bounds.GetBox().min << ", bounds max: " << bounds.GetBox().max  );
	DRD_LOG_INFO( L, (result ? "SUCCESS" : "FAIL") );
	return result;
}

//-------------------------------------------------------------------------------------------------
template< typename T >
void save( const DeviceVector<T>& i_Src, const std::string& i_path )
{
	HostVector<T> h;
	i_Src.getValue( h );

	std::ofstream outfile;
	outfile.open( i_path.c_str() );

	size_t n = i_Src.size();
	outfile << "size = " << n << std::endl;
	for( size_t i = 0; i < n; ++i )
	{
		outfile << h[i] << std::endl;
	}

	outfile.close();
}

// explicit instantiation
template void save<int>( const DeviceVector<int>& i_src, const std::string& i_path );
template void save<float>( const DeviceVector<float>& i_src, const std::string& i_path );
template void save<Imath::V2f>( const DeviceVector<Imath::V2f>& i_src, const std::string& i_path );
template void save<Imath::V3f>( const DeviceVector<Imath::V3f>& i_src, const std::string& i_path );
template void save<Imath::V4f>( const DeviceVector<Imath::V4f>& i_src, const std::string& i_path );


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
