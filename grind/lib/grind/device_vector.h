/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: $"
 * SVN_META_ID = "$Id: device_vector.h 99443 2011-08-30 01:14:50Z hugh.rayner $"
 */

#ifndef grind_device_vector_h
#define grind_device_vector_h

//! @cond DEV

//-------------------------------------------------------------------------------------------------
#include "log.h"
#include "context.h"
#include <pthread.h>
#include <stdexcept>

//#define LOG_DEVICE_VECTOR_ALLOC 1

#if 1
//TODO: get these outa here
#include <GL/glew.h>
#include <GL/glu.h>

#include <cuda_runtime_api.h>
#include <cutil.h>
#include <cutil_inline_runtime.h>
#include <cuda_gl_interop.h>
#else
// avoid pulling in GL/cuda headers etc
#endif

#include <vector>

namespace grind {

//-------------------------------------------------------------------------------------------------
// pre-declare
template< typename T >
class HostVector;

//-------------------------------------------------------------------------------------------------
//! an array of data on the device, drawable via OpenGL, processable via Cuda
template< typename T >
class DeviceVector
{
public:
	//! various states the device vector can be in
	enum State { DV_NONE, DV_OPEN_GL, DV_CUDA, DV_HOST };

	//! default constructor
	DeviceVector();

	//! construct with target and usage
	DeviceVector( unsigned int target //!< desired OpenGL target (set to zero if you don't want any OpenGL)
	            , unsigned int usage //!< OpenGL usage hint
	            );

	DeviceVector( const DeviceVector<T>& src );

	//! shared by the constructors
	void init();

	//! destructor
	virtual ~DeviceVector();

	//! is this device vector a valid one and is it ok to use?
	bool isOk() const;

	//! get the OpenGL id
	unsigned int getGLId() const;

	//! get the device ptr to do some cuda processing
	T* getDevicePtr() const;

	//! indicate the intended usage for the buffer
	void setBufferType( unsigned int target, unsigned int usage );

	//! setValue from host data
	void setValueHost( const T* i_Src, size_t i_Count );

	//! setValue from device data
	void setValueDevice( const T* i_Src, size_t i_Count );

	//! host -> device
	void setValue( const std::vector<T>& i_Src );

	//! device -> device
	void setValue( const DeviceVector<T>& i_Src );

	//! device -> device
	void setValue( const DeviceVector<T>& i_Src, size_t i_Count );

	//! assignment operator
	void operator=( const DeviceVector<T>& i_Src ){ setValue( i_Src ); }

	//! device -> host
	void getValue( std::vector<T>& o_Result ) const;

	//! device -> host
	void getValueAppend( std::vector<T>& o_Result ) const;

	//! device -> host
	void getSingleValue( unsigned a_Index, T & o_Result ) const;

	//! number of elements
	size_t size() const;

	//! memory requirements
	size_t memSize() const;

	//! clear all data
	void clear()
	{
		resize(0);
	}

	//! resize to this number of elements
	void resize( size_t sz );

	//! resize with a default value
	void resize( size_t sz, const T& i_Default );

	//! do appropriate glBindBuffer call
	void bindGL() const;

	//! do appropriate glBindBuffer reset call
	void unbindGL() const;

	//! indicate that we're about to do some cuda processing (optional)
	void prepForCuda() const;

	//! indicate that we're about to do some GL drawing (optional)
	void prepForGL() const;

	//! dump to log
	void dump() const;

	//! verbose dump to log
	void dumpVerbose() const;

	//! dump some info to log
	void dumpInfo() const;

	//! equivalence operator
	bool operator==( const DeviceVector<T>& other ) const;

	//! get the current state of the vector
	State getState(){ return m_State; }

private:
	//! pointer to device data for cuda processing
	T* m_DevicePtr;

	//! the gl buffer id
	unsigned int m_GLBuffer;

#if CUDART_VERSION >= 3000
	//! the cuda resource
	mutable cudaGraphicsResource* m_CudaGraphicsResource;
#endif

	//! is any device memory allocated?
	bool m_Allocated;

	//! size of the data
	size_t m_Size;

	//! intended OpenGL target
	unsigned int m_Target;

	//! intended OpenGL usage
	unsigned int m_Usage;

	void deallocate();
	void allocate( size_t sz );

	//! the current state of the data
	mutable State m_State;

	//! who allocated the date
	mutable State m_AllocatedBy;

	//! a stream for this devicevector
	cudaStream_t m_Stream;

	unsigned int original_thread_id;
};


//-------------------------------------------------------------------------------------------------------------------------------------
//! an easily bound python object for passing around references to internal DeviceVector<float> s
template< typename T >
struct DeviceVectorHandle
{
	DeviceVector<T>* _data;

	DeviceVectorHandle()
	: _data(NULL)
	{}

	DeviceVectorHandle( DeviceVector<T>* rhs )
	: _data( rhs )
	{}

	DeviceVector<T>& get() { return *_data; }
	const DeviceVector<T>& get() const { return *_data; }
};


//-------------------------------------------------------------------------------------------------------------------------------------
template< typename T >
DeviceVector<T>::DeviceVector()
{
	init();
}

//-------------------------------------------------------------------------------------------------
template< typename T >
DeviceVector<T>::DeviceVector( unsigned int target, unsigned int usage )
{
	init();
	m_Target = target;
	m_Usage = usage;
}

//-------------------------------------------------------------------------------------------------
template< typename T >
DeviceVector<T>::DeviceVector( const DeviceVector<T>& rhs )
{
	init();
	m_Target = rhs.m_Target;
	m_Usage = rhs.m_Usage;
	setValue( rhs );
}

//-------------------------------------------------------------------------------------------------
template< typename T >
void DeviceVector<T>::init()
{
	m_Allocated = false;
	m_Size = 0;
	m_DevicePtr = 0;
	m_Target = 0;
	m_Usage = 0;
	m_AllocatedBy = DV_NONE;
	m_State = DV_NONE;
	original_thread_id = ( unsigned int ) pthread_self();
}

//-------------------------------------------------------------------------------------------------
template< typename T >
DeviceVector< T >::~DeviceVector()
{
	deallocate();
}

//-------------------------------------------------------------------------------------------------
template< typename T >
bool
DeviceVector< T >::isOk() const
{
	return size() != 0;
}

//-------------------------------------------------------------------------------------------------
template< typename T >
void DeviceVector< T >::allocate( size_t sz )
{
	//grind::log << "Allocating DeviceVector in thread: " << ( unsigned int ) pthread_self() << ", from original thread: " << original_thread_id << "\n";

	if( m_Allocated ) deallocate();
	// leave de-allocated if the size is zero
	if( sz == 0 ) return;

	// if no gpu and no emulation then things are looking bad
	if( !ContextInfo::instance().hasGPU() && !ContextInfo::instance().hasEmulation() )
	{
		ContextInfo::instance().dump();
		throw std::runtime_error( "DeviceVector needs a gpu or emulation" );
	}

	if( ContextInfo::instance().hasOpenGL() && m_Target != 0 )
	{
		// create buffer object
		SAFE_GL( glGenBuffers( 1, &m_GLBuffer ) );

#if LOG_DEVICE_VECTOR_ALLOC
		DRD_LOG_DEBUG( L, "allocating DeviceVector (OpenGL/CUDA capable)[size=" << sz << "][bufferID=" << m_GLBuffer << "]" );
#endif

		SAFE_GL( glBindBuffer( m_Target, m_GLBuffer ) );

		// allocate buffer object
		SAFE_GL( glBufferData( m_Target, sz * sizeof(T), NULL, m_Usage ) );

		// but unbind it (should always be bound manually)
		SAFE_GL( glBindBuffer( m_Target, 0 ) );

		// register buffer object with CUDA
#if CUDART_VERSION < 3000
		SAFE_CUDA( cudaGLRegisterBufferObject( m_GLBuffer ) );
#else
		SAFE_CUDA( cudaGraphicsGLRegisterBuffer( &m_CudaGraphicsResource, m_GLBuffer, cudaGraphicsMapFlagsNone ) );
#endif

		m_State = DV_OPEN_GL;
		m_AllocatedBy = DV_OPEN_GL;

		checkErrorGL();
	}
	else
	{
		// straight cuda allocation
#ifdef __DEVICE_EMULATION__
#if LOG_DEVICE_VECTOR_ALLOC
		DRD_LOG_DEBUG( L, "allocating DeviceVector (host)" );
#endif
		m_DevicePtr = (T*)malloc( sz * sizeof(T) );
		m_State = DV_HOST;
		m_AllocatedBy = DV_HOST;
#else
#if LOG_DEVICE_VECTOR_ALLOC
		DRD_LOG_INFO( L, "allocating DeviceVector (CUDA capable)" );
#endif
		SAFE_CUDA( cudaMalloc( (void**) &m_DevicePtr, sz * sizeof(T) ) );
		m_State = DV_CUDA;
		m_AllocatedBy = DV_CUDA;
#endif
	}

	m_Size = sz;
	m_Allocated = true;
}


//-------------------------------------------------------------------------------------------------
template< typename T >
void DeviceVector< T >::deallocate()
{
#ifndef __DEVICE_EMULATION__
	// for cpu build this is not important
	if( ( unsigned int ) pthread_self() != original_thread_id ){
		DRD_LOG_DEBUG( L, "Deallocating DeviceVector in thread: " << ( unsigned int ) pthread_self() << ", from original thread: " << original_thread_id );
	}
#endif

	if ( m_Allocated )
	{
		if ( m_AllocatedBy == DV_OPEN_GL )
		{
#if LOG_DEVICE_VECTOR_ALLOC
			DRD_LOG_DEBUG( L, "de-allocating DeviceVector (OpenGL buffer id: " << m_GLBuffer << ")");
#endif

			if( !ContextInfo::instance().hasOpenGL() ){
				DRD_LOG_ERROR( L, "trying to deallocate device vector after gl context has gone" );
				glDeleteBuffers( 1, &m_GLBuffer );
				checkErrorGL();
				return;
			}

			bindGL();
#if 1

#if CUDART_VERSION < 3000
			SAFE_CUDA( cudaGLUnregisterBufferObject( m_GLBuffer ) );
#else
			SAFE_CUDA( cudaGraphicsUnregisterResource( m_CudaGraphicsResource ) );
#endif

			SAFE_GL( glDeleteBuffers( 1, &m_GLBuffer ) );


#else
			DRD_LOG_INFO( L, "warning: deallocation of device vector disabled pending response from nvidia" );

			// reallocate with zero size (doesn't seem to actually release memory but left in for now)
			SAFE_GL( glBufferDataBEE( m_Target, 0, NULL, m_Usage ) );

#endif
			m_GLBuffer = 0;

			checkErrorGL();
		}
		else
		{
#ifdef __DEVICE_EMULATION__
#if LOG_DEVICE_VECTOR_ALLOC
			DRD_LOG_DEBUG( L, "de-allocating DeviceVector (host)" );
#endif
			assert( m_AllocatedBy == DV_HOST );
			free( m_DevicePtr );
#else
#if LOG_DEVICE_VECTOR_ALLOC
			DRD_LOG_DEBUG( L, "de-allocating DeviceVector (CUDA capable)" );
#endif
			// if allocated by cuda we need to deallocate in the same thread
			assert( ( unsigned int ) pthread_self() == original_thread_id );
			SAFE_CUDA( cudaFree( m_DevicePtr ) );
#endif
			m_DevicePtr = NULL;
		}
	}
	m_Allocated = false;
	m_Size = 0;
}


//-------------------------------------------------------------------------------------------------
template< typename T >
void DeviceVector< T >::setBufferType( 	unsigned int target,
										unsigned int usage )
{
	m_Target = target;
	m_Usage = usage;
}

//-------------------------------------------------------------------------------------------------
template< typename T >
size_t DeviceVector< T >::size() const
{
	return m_Size;
}

//-------------------------------------------------------------------------------------------------
template< typename T >
size_t DeviceVector< T >::memSize() const
{
	return m_Size * sizeof(T);
}

//-------------------------------------------------------------------------------------------------
template< typename T >
void DeviceVector< T >::resize( size_t sz )
{
	if ( sz != m_Size )
	{
		allocate( sz );
	}
}

//-------------------------------------------------------------------------------------------------
template< typename T >
void DeviceVector< T >::resize( size_t sz, const T& i_Default )
{
	if ( sz == m_Size ) return;

	//TODO: suboptimal for now
	HostVector< T > temp;
	temp.resize( sz, i_Default );

	setValue( temp );
}


//-------------------------------------------------------------------------------------------------
template< typename T >
unsigned int DeviceVector< T >::getGLId() const
{
	if( m_AllocatedBy != DV_OPEN_GL )
		throw std::runtime_error( "can't get GL Id for DeviceVector not allocated by GL" );

	if ( m_State != DV_OPEN_GL ) prepForGL();

	return m_GLBuffer;
}

//-------------------------------------------------------------------------------------------------
template< typename T >
T* DeviceVector< T >::getDevicePtr() const
{
	if( m_State != DV_CUDA )
	{
		prepForCuda();
	}

	return m_DevicePtr;
}

//-------------------------------------------------------------------------------------------------
template< typename T >
void DeviceVector< T >::setValueHost( const T* i_Src, size_t i_Count )
{
	// make sure our size is correct
	if( m_Size != i_Count )
	{
		resize( i_Count );
	}
	if( i_Count == 0 )
	{
		return;
	}
	prepForCuda();
	assert( m_DevicePtr );
	assert( &i_Src[ 0 ] );
	assert( m_Size );

#ifdef __DEVICE_EMULATION__
	memcpy( m_DevicePtr, i_Src, m_Size * sizeof(T) );
#else
	SAFE_CUDA( cudaMemcpy( m_DevicePtr, i_Src, m_Size * sizeof(T), cudaMemcpyHostToDevice ) );
#endif
}

//-------------------------------------------------------------------------------------------------
template< typename T >
void DeviceVector< T >::setValueDevice( const T* i_Src, size_t i_Count )
{
	// make sure our size is correct
	if( m_Size != i_Count )
	{
		resize( i_Count );
	}
	if( i_Count == 0 )
	{
		return;
	}
	prepForCuda();
	assert( m_DevicePtr );
	assert( &i_Src[ 0 ] );
	assert( m_Size );

#ifdef __DEVICE_EMULATION__
	memcpy( m_DevicePtr, i_Src, m_Size * sizeof(T) );
#else
	SAFE_CUDA( cudaMemcpy( m_DevicePtr, i_Src, m_Size * sizeof(T), cudaMemcpyDeviceToDevice ) );
#endif
}


//-------------------------------------------------------------------------------------------------
template< typename T >
void DeviceVector< T >::setValue( const std::vector< T >& i_Src )
{
	setValueHost( &i_Src[0], i_Src.size() );
}

//-------------------------------------------------------------------------------------------------
template< typename T >
void DeviceVector< T >::setValue( const DeviceVector< T >& i_Src )
{
	setValueDevice( i_Src.getDevicePtr(), i_Src.size() );
}

//-------------------------------------------------------------------------------------------------
template< typename T >
void DeviceVector< T >::setValue( const DeviceVector< T >& i_Src, size_t n )
{
	assert( n <= i_Src.size() );
	setValueDevice( i_Src.getDevicePtr(), n );
}

//-------------------------------------------------------------------------------------------------
template< typename T >
void DeviceVector< T >::getValue( std::vector< T >& o_Result ) const
{
	// make sure destination is the correct size
	o_Result.resize( m_Size );
	if ( m_Size == 0 ) return;
	prepForCuda();
#ifdef __DEVICE_EMULATION__
	memcpy( &o_Result[ 0 ], m_DevicePtr, m_Size * sizeof(T) );
#else
	SAFE_CUDA( cudaMemcpy( &o_Result[ 0 ], m_DevicePtr, m_Size * sizeof(T), cudaMemcpyDeviceToHost ) );
#endif
}

//-------------------------------------------------------------------------------------------------
template< typename T >
void DeviceVector< T >::getValueAppend( std::vector< T >& o_Result ) const
{
	int oldSize = o_Result.size();
	// make sure destination is the correct size
	o_Result.resize( oldSize + m_Size );
	if ( m_Size == 0 ) return;
	prepForCuda();
#ifdef __DEVICE_EMULATION__
	memcpy( &o_Result[ oldSize ], m_DevicePtr, m_Size * sizeof(T) );
#else
	SAFE_CUDA( cudaMemcpy( &o_Result[ oldSize ], m_DevicePtr, m_Size * sizeof(T), cudaMemcpyDeviceToHost ) );
#endif
}

//-------------------------------------------------------------------------------------------------
template< typename T >
void DeviceVector< T >::getSingleValue( unsigned a_Index, T & o_Result ) const
{
	// make sure destination is the correct size
	prepForCuda();
#ifdef __DEVICE_EMULATION__
	memcpy( &o_Result, m_DevicePtr + a_Index, sizeof(T) );
#else
	SAFE_CUDA( cudaMemcpy( &o_Result, m_DevicePtr + a_Index, sizeof(T), cudaMemcpyDeviceToHost ) );
#endif
}

//-------------------------------------------------------------------------------------------------
template< typename T >
void DeviceVector< T >::bindGL() const
{
	if ( m_State != DV_OPEN_GL ) prepForGL();
	SAFE_GL( glBindBuffer( m_Target, m_GLBuffer ) );
}

//-------------------------------------------------------------------------------------------------
template< typename T >
void DeviceVector< T >::unbindGL() const
{
	if ( m_State != DV_CUDA ) prepForCuda();
	SAFE_GL( glBindBuffer( m_Target, 0 ) );
}

//-------------------------------------------------------------------------------------------------
template< typename T >
void DeviceVector< T >::prepForCuda() const
{
	if ( m_State != DV_OPEN_GL ) return;

	// DRD_LOG_DEBUG( L, "prepForCuda() << " << m_GLBuffer );
#if CUDART_VERSION < 3000
	SAFE_CUDA( cudaGLMapBufferObject( (void**) &m_DevicePtr, m_GLBuffer ) );
#else
	size_t num_bytes;
	SAFE_CUDA( cudaGraphicsMapResources( 1, &m_CudaGraphicsResource, 0 ) );
	SAFE_CUDA( cudaGraphicsResourceGetMappedPointer( (void**) &m_DevicePtr, &num_bytes, m_CudaGraphicsResource ) );
#endif
	// DRD_LOG_DEBUG( L, "prepForCuda() >>  " << m_DevicePtr << ":" << m_GLBuffer );

	m_State = DV_CUDA;
}

//-------------------------------------------------------------------------------------------------
template< typename T >
void DeviceVector< T >::prepForGL() const
{
	if( m_AllocatedBy != DV_OPEN_GL ){
		throw std::runtime_error( "can't prepForGL() when device vector wasn't allocated by GL" );
	}

	if ( !ContextInfo::instance().hasOpenGL() )
	{
		ContextInfo::instance().dump();
		throw std::runtime_error( "DeviceVector needs OpenGL context for any OpenGL operation" );
	}

	if ( m_State == DV_OPEN_GL ) return;

	// grind::spam << "prepForGL()" << std::endl;
#if CUDART_VERSION < 3000
	SAFE_CUDA( cudaGLUnmapBufferObject( m_GLBuffer ) );
#else
	SAFE_CUDA( cudaGraphicsUnmapResources( 1, &m_CudaGraphicsResource, 0 ) );
#endif
	m_State = DV_OPEN_GL;
}

//-------------------------------------------------------------------------------------------------
template< typename T >
void DeviceVector< T >::dump() const
{
	HostVector< T > v;
	getValue( v );
	v.dump();
}

//-------------------------------------------------------------------------------------------------
template< typename T >
void DeviceVector< T >::dumpVerbose() const
{
	HostVector< T > v;
	getValue( v );

	for ( int i = 0 ; i < v.size() ; ++i )
	{
		std::cerr << '[' << i << "]:" << v[ i ] << ", ";
	}
	std::cerr << "\n";
}

//-------------------------------------------------------------------------------------------------
template< typename T >
void DeviceVector< T >::dumpInfo() const
{
	DRD_LOG_INFO( L, "DeviceVector info:" );
	DRD_LOG_INFO( L, "\tsize: " << size() );
	DRD_LOG_INFO( L, "\tallocated by: " );
	switch( m_AllocatedBy ){
		case DV_NONE: DRD_LOG_INFO( L, "DV_NONE" ); break;
		case DV_OPEN_GL: DRD_LOG_INFO( L, "DV_OPEN_GL" ); break;
		case DV_CUDA: DRD_LOG_INFO( L, "DV_CUDA" ); break;
		case DV_HOST: DRD_LOG_INFO( L, "DV_HOST" ); break;
	}
	DRD_LOG_INFO( L, "\tstate: ");
	switch( m_State ){
		case DV_NONE: DRD_LOG_INFO( L, "DV_NONE" ); break;
		case DV_OPEN_GL: DRD_LOG_INFO( L, "DV_OPEN_GL" ); break;
		case DV_CUDA: DRD_LOG_INFO( L, "DV_CUDA" ); break;
		case DV_HOST: DRD_LOG_INFO( L, "DV_HOST" ); break;
	}
}

//-------------------------------------------------------------------------------------------------
template< typename T >
bool DeviceVector< T >::operator==( const DeviceVector<T>& rhs ) const
{
	if( rhs.size() != size() ) return false;

	// for now, host code
	HostVector< T > h_this, h_rhs;
	getValue( h_this );
	rhs.getValue( h_rhs );
	return std::equal( h_this.begin(), h_this.end(), h_rhs.begin() );
}



} // namespace grind

//! @endcond

#endif /* grind_device_vector_h */


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
