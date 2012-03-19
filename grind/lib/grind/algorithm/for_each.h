/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: $"
 * SVN_META_ID = "$Id: for_each.h 42733 2010-08-18 06:12:21Z allan.johns $"
 */

#ifndef grind_for_each_h
#define grind_for_each_h

#include <thrust/distance.h>
#include <thrust/for_each.h>

namespace grind
{

#ifdef __DEVICE_EMULATION__

//! CPU OpenMP parallel for_each
//! note that this is done via a macro as NVCC didn't seem to like a template expansion (opmp parallel for provided no acceleration)
#define GRIND_FOR_EACH( f, l, op ) \
{ \
	int n = thrust::distance(f,l); \
	_Pragma( "omp parallel for" ) \
	for( int i = 0; i < n; ++i ) \
	{ \
		op( *(f+i) ); \
	} \
}

#else /* __DEVICE_EMULATION__ */

#ifdef __SUPPORT_SM_10__

//! round up integer division
inline int i_div_up( int a, int b ){
	return (a % b != 0) ? (a / b + 1) : (a / b);
}

//! kernel to call functor on each element
template< typename F >
__global__
void grindForEachKernel( int n, F f )
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if ( i >= n ) return;
	f(i);
}

//!
template< typename IT, typename F >
void grindForEach( IT first, IT last, F f )
{
	int n = last-first;

	dim3 block( 128, 1, 1 );
	dim3 grid( grind::i_div_up(n, block.x), 1, 1 );

	grindForEachKernel<<<grid, block>>>( n, f );
}


//! Device mode was using thrust::for_each, switched to custom implementation for compatibility with old cards
#define GRIND_FOR_EACH( f, l, op ) \
{ \
	grind::grindForEach( f, l, op ); \
}

#else /* __SUPPORT_SM_10__ */

#define GRIND_FOR_EACH( f, l, op ) \
{ \
	thrust::for_each( f, l, op ); \
}

#endif /* __SUPPORT_SM_10__ */
#endif /* __DEVICE_EMULATION__ */


} // grind namespace

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
