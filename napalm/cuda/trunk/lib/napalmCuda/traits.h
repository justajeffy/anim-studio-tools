
#include "cutil_math.h"
#include <OpenEXR/half.h>

namespace napalm
{

	//! Traits class that maps cpu and gpu types
	template< typename T >
	struct CudaTraits
	{
		// default implementation has matching types
		typedef T cpu_value_type;
		typedef T cuda_value_type;
	};

	//! Specialization for half
	template<>
	struct CudaTraits< half >
	{
		typedef half cpu_value_type;
		typedef unsigned short cuda_value_type;
	};

	//! Specialization for V3f
	template<>
	struct CudaTraits< Imath::V3f >
	{
		typedef Imath::V3f cpu_value_type;
		typedef float3 cuda_value_type;
	};

	// type conversions

	template< typename T >
	struct to_cuda_type
	{
		static typename CudaTraits< T >::cuda_value_type value( const T& v )
		{
			return v;
		}
	};

	template<>
	struct to_cuda_type< Imath::V3f >
	{
		static float3 value( const Imath::V3f& v )
		{
			return make_float3(v.x,v.y,v.z);
		}
	};


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
