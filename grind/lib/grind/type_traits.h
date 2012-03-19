/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: $"
 * SVN_META_ID = "$Id: type_traits.h 42544 2010-08-17 04:31:03Z allan.johns $"
 */

#ifndef grind_type_traits_h
#define grind_type_traits_h


// cuda types
#include <vector_types.h>
#include <OpenEXR/ImathVec.h>

namespace grind {

	//-------------------------------------------------------------------------------------------------
	//! type traits to map between drd types and cuda types
	template< typename T >
	struct type_traits
	{
	};

	//-------------------------------------------------------------------------------------------------
	template< >
	struct type_traits< Imath::V3f >
	{
		typedef Imath::V3f grind_type;
		typedef float3 cuda_type;
	};

	//-------------------------------------------------------------------------------------------------
	template< >
	struct type_traits< Imath::V2f >
	{
		typedef Imath::V2f grind_type;
		typedef float2 cuda_type;
	};

	//-------------------------------------------------------------------------------------------------
	template< >
	struct type_traits< float >
	{
		typedef float grind_type;
		typedef float cuda_type;
	};
}

#endif /* grind_type_traits_h */


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
