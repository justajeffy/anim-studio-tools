/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: $"
 * SVN_META_ID = "$Id: atomic.h 80224 2011-05-05 05:13:49Z chris.cooper $"
 */

#ifndef grind_atomic_h
#define grind_atomic_h

#include <thrust/distance.h>
#include <thrust/for_each.h>
#include <cutil_math.h>

namespace grind
{

// allow atomic operations in the cpu code
#define USE_CPU_ATOMIC 1

#if USE_CPU_ATOMIC
// openmp parallel for each
#define GRIND_FOR_EACH_ATOMIC( f, l, op ) GRIND_FOR_EACH( f, l, op );
#else
// serial for each
#define GRIND_FOR_EACH_ATOMIC( f, l, op ) thrust::for_each( f, l, op );
#endif


#ifdef __DEVICE_EMULATION__
#define GRIND_HOST_DEVICE_ATOMIC __host__
#else
#define GRIND_HOST_DEVICE_ATOMIC __device__
#endif


#ifdef __DEVICE_EMULATION__

#if USE_CPU_ATOMIC
// should be correct OpenMP atomic usage.
// Note that this has to be a define otherwise the pragma doesn't appear to work
#define atomicAddT(INC,RESULT) \
{ \
	_Pragma( "omp atomic" ) \
	RESULT += INC; \
}
#else
// conventional add, relying on loop being serial
#define atomicAddT(INC,RESULT) \
{ \
	RESULT += INC; \
}

#endif

#else
// http://forums.nvidia.com/index.php?showtopic=67691&pid=380935&mode=threaded&start=#entry380935
__device__
inline void atomicFloatAdd(float *address, float val)
{
	int i_val = __float_as_int(val);
	int tmp0 = 0;
	int tmp1;

	while( (tmp1 = atomicCAS((int *)address, tmp0, i_val)) != tmp0)
	{
		tmp0 = tmp1;
		i_val = __float_as_int(val + __int_as_float(tmp1));
	}
}

//! atomic add of a float3
//! TODO: may be able to lock just on x element
__device__
inline void atomicAddT( const float3& inc, float3& result )
{
    atomicFloatAdd( &result.x, inc.x );
    atomicFloatAdd( &result.y, inc.y );
    atomicFloatAdd( &result.z, inc.z );
}

__device__
inline void atomicAddT( const float2& inc, float2& result )
{
    atomicFloatAdd( &result.x, inc.x );
    atomicFloatAdd( &result.y, inc.y );
}

__device__
inline void atomicAddT( const float& inc, float& result )
{
    atomicFloatAdd( &result, inc );
}

__device__
inline void atomicAddT( const int& inc, int& result )
{
	// cuda atomic integer add
	atomicAdd( &result, inc );
}


#endif

} // grind namespace

#endif /* grind_atomic_h */


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
