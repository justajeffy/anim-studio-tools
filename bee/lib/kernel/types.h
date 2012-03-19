/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: http://svn/drd/apps/bee/trunk/lib/kernel/types.h $"
 * SVN_META_ID = "$Id: types.h 22841 2010-02-16 07:17:08Z stephane.bertout $"
 */

//-------------------------------------------------------------------------------------------------
#ifndef bee_types_h
#define bee_types_h
#pragma once

#undef Bool // thanks GLEW...

namespace bee
{
	// 64b Signed Integer Type
	typedef long long			Int64;
	//! 64b Unsigned Integer Type
	typedef unsigned long long	Uin64;

	//! 32b Unsigned Integer Type
	typedef unsigned int 		UInt;
	//! 32b Signed Integer Type
	typedef signed int 			Int;

	//! 32b Unsigned Integer Type
	typedef unsigned int 		UInt32;
	//! 32b Signed Integer Type
	typedef signed int 			Int32;
	//! 64b Unsigned Integer Type
	typedef unsigned long long	UInt64;
	//! 64b Signed Integer Type
	typedef long long			Int64;

	//! Unsigned Short Type (16b)
	typedef unsigned short 		UShort;
	//! Signed Short Type (16b)
	typedef signed short 		Short;

	//! Unsigned Char Type (8b)
	typedef unsigned char 		UChar;
	//! Signed Char Type (8b)
	typedef signed char 		Char;

	//! Float Type (32b)
	typedef float 				Float;
	//! Double Type (64b)
	typedef double 				Double;

	//! Boolean Type
	typedef bool 				Bool;

	//typedef float 				Time;
	// typedef size_t				Size;

	//-------------------------------------------------------------------------------------------------
	#ifndef NULL
		#define NULL 0
	#endif

	//-------------------------------------------------------------------------------------------------
	class Face
	{
	public:
		UInt a,b,c;
		Face(){}
		Face( UInt a_A, UInt a_B, UInt a_C )
		: a(a_A), b(a_B), c(a_C) {}
	};

	//-------------------------------------------------------------------------------------------------
	#define ATTRIB(TYPE,NAME) 																	\
		public: inline const TYPE & Get##NAME() const { return m_##NAME; }						\
		public: inline       TYPE & Get##NAME()       { return m_##NAME; }						\
		public: inline      void Set##NAME( const TYPE & a_##NAME ) { m_##NAME = a_##NAME; }	\
		protected: TYPE m_##NAME;

	#include <boost/assert.hpp>
	#define ASSERT BOOST_ASSERT
}

#endif // bee_types_h


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
