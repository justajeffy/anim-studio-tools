#ifndef _NAPALM_CUDA_BUFFERSTORECUDA__H_
#define _NAPALM_CUDA_BUFFERSTORECUDA__H_

// napalm includes
#include <napalm/core/BufferStore.h>
#include <napalm/core/util/fixed_range.hpp>

// our includes
#include "napalmCuda/traits.h"
#include "napalmCuda/DeviceVector.h"


// allow logging of host->device and device->host transfers
#ifndef NAPALM_LOG_CPU_GPU_TRANSFERS
#define NAPALM_LOG_CPU_GPU_TRANSFERS 0
#else
#include <iostream>
#endif

// using just napalm namespace as this is an extension of core napalm functionality
namespace napalm
{

/*
 * @class BufferStoreCuda
 * @brief
 * A gpu buffer
 */
template< typename T >
class BufferStoreCuda: public BufferStore
{
public:

	typedef T											value_type;
	typedef typename CudaTraits<T>::cuda_value_type		cuda_value_type;
	typedef DeviceVector< cuda_value_type >				cuda_buffer_type;

	BufferStoreCuda( 	unsigned int size = 0, const T& value = T() );

	virtual unsigned int size() const{ return m_data.size(); }
	virtual bool resize(unsigned int n, bool destructive = false);

	virtual bool copyTo( store_ptr store ) const;
	virtual bool copyFrom( c_store_ptr store );
	virtual store_ptr createSerializableCopy();
	virtual void shrink();

	//! Const access to buffer for client code
	const cuda_buffer_type& buffer() const { return m_data; }

	//! Non-const access to buffer for client code
	cuda_buffer_type& buffer() { return m_data; }

protected:

	// CPU storage convenience types
	typedef util::fixed_range<T*>					w_type;
	typedef util::fixed_range<const T*>				r_type;

	//! Underlying data
	cuda_buffer_type m_data;
};

///////////////////////// impl

template< typename T >
BufferStoreCuda< T >::BufferStoreCuda( 	unsigned int size,
											const T& value ) :
	BufferStore( false, false, false, true, false, false ), m_data( size, to_cuda_type<T>::value(value) )
{
}

template< typename T >
bool BufferStoreCuda< T >::copyTo( store_ptr destStore ) const
{
	assert( destStore );

	if(!destStore->writable())
		return false;

	if(!destStore->resize(m_data.size()))
		return false;

	w_type frDest = destStore->w<T>();
	assert( frDest.size() == m_data.size() );

#if NAPALM_LOG_CPU_GPU_TRANSFERS
	std::cout << "## device->host transfer ##" << std::endl;
#endif
	// we're copying between different binary compatible types
	cudaMemcpy( frDest.begin(), thrust::raw_pointer_cast( &m_data[ 0 ] ), m_data.size() * sizeof(T), cudaMemcpyDeviceToHost );

	return true;
}

template< typename T >
bool BufferStoreCuda< T >::copyFrom( c_store_ptr srcStore )
{
	assert( srcStore );

	if(!srcStore->readable())
		return false;

	r_type frSrc = srcStore->r<T>();

	m_data.resize(srcStore->size());

#if NAPALM_LOG_CPU_GPU_TRANSFERS
	std::cout << "## host->device transfer ##" << std::endl;
#endif
	// we're copying between different binary compatible types
	cudaMemcpy( thrust::raw_pointer_cast( &m_data[ 0 ] ), frSrc.begin(), frSrc.size() * sizeof(T), cudaMemcpyHostToDevice );

	return true;
}

template<typename T>
store_ptr BufferStoreCuda<T>::createSerializableCopy()
{
#if 0
	// if the data for this store is still on disk, then return a clone of us for
	// serialising - that way the data will be loaded temporarily for serialisation,
	// then deallocated.
	if(m_data.isLoaded())
		return boost::static_pointer_cast<BufferStore>(shared_from_this());
	else
		return store_ptr(new BufferStoreCpu<T>(*this));
#endif
	throw std::runtime_error( "not implemented" );
}

template<typename T>
bool BufferStoreCuda<T>::resize(unsigned int n, bool destructive)
{
	m_data.resize(n);
	return true;
}

template<typename T>
void BufferStoreCuda<T>::shrink()
{
	std::cout << "# not supported yet" << std::endl;
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
