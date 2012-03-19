#ifndef _NAPALM_CUDA_DEVICE_VECTOR__H_
#define _NAPALM_CUDA_DEVICE_VECTOR__H_

#include <thrust/device_ptr.h>

// log allocations and de-allocations
#ifndef NAPALM_LOG_GPU_ALLOC
#define NAPALM_LOG_GPU_ALLOC 0
#endif

namespace napalm
{

cudaError_t safeCudaMalloc( void ** devPtr, size_t sz );
cudaError_t safeCudaFree( void* devPtr );

/*!
 *	DeviceVector class
 * \note: this class exists because of problems compiling thrust::device_vector with non standard types on older gcc versions
 */
template< typename T >
struct DeviceVector
{
	typedef thrust::device_ptr<T> iterator;
	typedef const thrust::device_ptr<T> const_iterator;
	typedef thrust::device_reference<T> reference;
	typedef const thrust::device_reference<T> const_reference;

	//! Create an empty buffer
	DeviceVector()
	{
		init();
	}

	//! Create a buffer of a certain size with each element initialized to value
	DeviceVector( size_t sz, const T& value = T() )
	{
		init();
		resize( sz );
		fill( value );
	}

	//! Fill every element of this DeviceVector with a value
	void fill( const T& value );

	//! Destructor
	~DeviceVector()
	{
		deallocate();
	}

	//! Resize this DeviceVector
	void resize( size_t sz )
	{
		if( sz == m_elementCount ) return;
		deallocate();
		if( sz == 0 ) return;

		allocate( sz );
	}

	reference operator[]( size_t os )
	{
		return *(begin()+os);
	}

	const_reference operator[]( size_t os ) const
	{
		return *(begin()+os);
	}

	//! return the number of elements
	size_t size() const
	{
		return m_elementCount;
	}

	//! memory size in bytes
	size_t memSize() const
	{
		return m_elementCount * sizeof(T);
	}

	iterator begin(){ return iterator(m_buffer); }
	iterator end() { return iterator(m_buffer + memSize() ); }

	const_iterator begin() const { return iterator(m_buffer); }
	const_iterator end() const { return iterator(m_buffer + memSize() ); }

protected:

	void init()
	{
		m_buffer = NULL;
		m_elementCount = 0;
	}

	void allocate( size_t sz )
	{
#if NAPALM_LOG_GPU_ALLOC
		std::cerr << "## allocating " << sizeof(T) * sz << " bytes on device (thread id: " << ( unsigned int ) pthread_self() << ") ##" << std::endl;
#endif
		// need to do this on a temp variable to get around type-punning warning
		void* new_data;
		safeCudaMalloc( (void**) ( &new_data ), sizeof(T) * sz );
		m_buffer = reinterpret_cast< T* > ( new_data );

		if( m_buffer ){
			m_elementCount = sz;
		} else {
			std::cerr << "ERROR trying to allocate" << std::endl;
		}
	}

	void deallocate()
	{
		if( m_buffer ){
#if NAPALM_LOG_GPU_ALLOC
			std::cerr << "## deallocating (thread id: " << ( unsigned int ) pthread_self() << ") ##" << std::endl;
#endif
			safeCudaFree( m_buffer );
		}

		m_buffer = NULL;
		m_elementCount = 0;
	}

	//! Actual storage
	T* m_buffer;

	//! Number of elements stored
	size_t m_elementCount;
};


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
