/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: $"
 * SVN_META_ID = "$Id: cl_vector.h 42544 2010-08-17 04:31:03Z allan.johns $"
 */

#ifndef grind_cl_vector_h
#define grind_cl_vector_h

//-------------------------------------------------------------------------------------------------
#include <boost/scoped_ptr.hpp>
#include <boost/utility.hpp>

#include <GL/gl3.h>
#include <CL/cl.h>
#include <CL/cl_gl.h>

namespace grind {

//! @cond DEV

//-------------------------------------------------------------------------------------------------
//! for managing <code> mapping of the buffer object via scope
template< typename T >
class device_ptr_handle : boost::noncopyable
{
	GLuint _bufferCL;
	GLuint _bufferGL;

public:
	device_ptr_handle( GLuint bufferGL, GLuint bufferCL )
	: _bufferGL( bufferGL )
	, _bufferCL( bufferCL )
	{
		cl_int ciErrNum = CL_SUCCESS;
		ciErrNum |= clEnqueueAcquireGLObjects(cqCommandQue, 1, &_bufferCL, 0,0,0);
		//cutilSafeCall( cudaGLMapBufferObject((void**)&_device_ptr, _bufferGL) );
	}

	~device_ptr_handle()
	{
		//cutilSafeCall( cudaGLUnmapBufferObject(_bufferGL) );
		cl_int ciErrNum = CL_SUCCESS;
		ciErrNum |= clEnqueueReleaseGLObjects(cqCommandQue, 1, &_bufferCL, 0,0,0);
	}

	T* device_ptr() { return _device_ptr; }
	const T* device_ptr() const { return _device_ptr; }
};


//-------------------------------------------------------------------------------------------------
//! an OpenGL buffer object with managed memory that is editable by cuda
template< typename T, GLenum TARGET, GLenum USAGE >
class buffer_vector : boost::noncopyable
{

	GLuint _bufferGL;
	GLuint _bufferCL;

	size_t _sz;
	bool _allocated;

	void deallocate()
	{
		if( _allocated ){
			glBindBuffer(1, _bufferGL);
			glDeleteBuffers(1, &_bufferGL);

			_bufferGL = 0;
			_bufferCL = 0;
		}
		_allocated = false;
	}

	void allocate( size_t sz )
	{
		if( _allocated ) deallocate();

		std::cerr << "resizing to " << sz << std::endl;

		// create buffer object
		glGenBuffers(1, &_bufferGL);
		glBindBuffer( TARGET, _bufferGL);

		// initialize buffer object
		glBufferData( TARGET, sz*sizeof(T), 0, USAGE );
		glBindBuffer( TARGET, 0 );

		// register buffer object with CUDA
//		cutilSafeCall( cudaGLRegisterBufferObject( _bufferGL ) );
		_bufferCL = clCreateFromGLBuffer( cxGPUContext, CL_MEM_WRITE_ONLY, *_bufferGL, NULL );

		_sz = sz;
		_allocated = true;
	}

public:

	typedef boost::scoped_ptr< device_ptr_handle<T> > handle_type;

	buffer_vector()
	: _allocated(false)
	, _sz(0)
	{
		//allocate(sz);
	}

	~buffer_vector() { deallocate(); }

	size_t size() const { return _sz; }
	size_t mem_size() const { return _sz * sizeof(T); }

	void resize( size_t sz ) { if( sz != _sz ) allocate(sz); }

	void get_device_ptr_handle( handle_type& handle )
	{
		handle.reset( new device_ptr_handle<T>(_bufferGL) );
	}

	GLuint get_buffer() const { return _bufferGL; }

	void operator=( const std::vector<T>& rhs )
	{
		if( _sz != rhs.size() ) resize( rhs.size() );
		glBindBuffer( TARGET, _bufferGL);
		T* data = (T*)glMapBuffer( TARGET, GL_WRITE_ONLY );
		if( data == NULL ){ std::cerr << "unable to map buffer"; }
		else {
			std::copy( rhs.begin(), rhs.end(), data );
			glUnmapBuffer( TARGET );
		}
		glBindBuffer( TARGET, 0 );
	}
};

//! @endcond

} // grind

#endif /* grind_cl_vector_h */


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
