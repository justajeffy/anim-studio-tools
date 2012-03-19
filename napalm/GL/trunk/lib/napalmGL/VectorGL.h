#ifndef _NAPALM_GL_GL_VECTOR__H_
#define _NAPALM_GL_GL_VECTOR__H_

#include "context.h"
#include <iostream>
#include <cassert>
#include <GL/glew.h>
#include <GL/glu.h>

namespace napalm_gl
{

template< typename T >
struct VectorGL
{
	VectorGL( 	unsigned int target,
				unsigned int usage );
	~VectorGL();

	void setBufferType( unsigned int target,
						unsigned int usage );

	size_t size() const;

	void resize( size_t sz );

private:

	void init();
	void allocate( size_t sz );
	void deallocate();

	//! intended OpenGL target
	unsigned int m_Target;

	//! intended OpenGL usage
	unsigned int m_Usage;

	bool m_Allocated;
	size_t m_Size;
	unsigned int m_GLBuffer;
};

///////////////////////// impl

template< typename T >
VectorGL< T >::VectorGL( 	unsigned int target = GL_ARRAY_BUFFER,
							unsigned int usage = GL_DYNAMIC_DRAW )
{
	init();
	m_Target = target;
	m_Usage = usage;
}

template< typename T >
VectorGL< T >::~VectorGL()
{
	deallocate();
}

template< typename T >
void VectorGL< T >::init()
{
	m_Target = 0;
	m_Usage = 0;
	m_Allocated = false;
	m_Size = 0;
}

template< typename T >
void VectorGL< T >::allocate( size_t sz )
{
	if ( m_Allocated ) deallocate();
	if ( sz == 0 ) return;

	assert( hasOpenGL());
	// make sure glewInit() has been called
	assert( glGenBuffers != NULL );

#ifndef NDEBUG
	std::cout << "allocating OpenGL buffer of size: " << sz * sizeof(T) << std::endl;
#endif

	glGenBuffers( 1, &m_GLBuffer );
	glBindBuffer( m_Target, m_GLBuffer );
	glBufferData( m_Target, sz * sizeof(T), NULL, m_Usage );
	glBindBuffer( m_Target, 0 );

	assert( noErrorsGL());

	m_Size = sz;
	m_Allocated = true;
}

template< typename T >
void VectorGL< T >::deallocate()
{
	if ( !m_Allocated ) return;

	assert( hasOpenGL());

	glDeleteBuffers( 1, &m_GLBuffer );
	m_GLBuffer = 0;

	assert( noErrorsGL());
}

template< typename T >
void VectorGL< T >::setBufferType( 	unsigned int target,
									unsigned int usage )
{
	m_Target = target;
	m_Usage = usage;
}


template< typename T >
size_t VectorGL< T >::size() const
{
	return m_Size;
}

template< typename T >
void VectorGL< T >::resize( size_t sz )
{
	if ( sz == m_Size ) return;
	allocate( sz );
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
