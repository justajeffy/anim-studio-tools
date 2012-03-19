/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: $"
 * SVN_META_ID = "$Id: transform.h 42733 2010-08-18 06:12:21Z allan.johns $"
 */

#ifndef grind_transform_h
#define grind_transform_h

#include <thrust/distance.h>
#include <thrust/transform.h>

namespace grind
{

#ifdef __DEVICE_EMULATION__

//! CPU OpenMP parallel transform
//! note that this is done via a macro as NVCC didn't seem to like a template expansion (opmp parallel for provided no acceleration)
#define GRIND_TRANSFORM( f, l, rf, op ) \
{ \
	int n = thrust::distance(f,l); \
	_Pragma( "omp parallel for" ) \
	for( int i = 0; i < n; ++i ) \
	{ \
		*(rf+i) = op( *(f+i) ); \
	} \
}

#else

//! Device mode use thrust::transform
#define GRIND_TRANSFORM( f, l, rf, op ) \
{ \
	thrust::transform( f, l, rf, op ); \
}

#endif

}

#endif /* grind_transform_h */



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
