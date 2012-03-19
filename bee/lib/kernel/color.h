/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: http://svn/drd/apps/bee/trunk/lib/kernel/color.h $"
 * SVN_META_ID = "$Id: color.h 27186 2010-04-07 01:35:04Z david.morris $"
 */

#ifndef bee_color_h
#define bee_color_h
#pragma once

#include "../kernel/types.h"

namespace bee
{
	//-------------------------------------------------------------------------------------------------
	//! Colour is a KERNEL utility class encapsulating a simple RGBA Float colour
	class Colour
	{
	public:
		//! Red Float Component
		Float r;
		//! Green Float Component
		Float g;
		//! Blue Float Component
		Float b;
		//! Alpha Float Component
		Float a;

		//! Default Constructor (no initialisation)
		Colour()
		{
		}
		//! RGBA Constructor
		Colour( Float a_R, Float a_G, Float a_B, Float a_A = 1.0f ) :
			r( a_R ), g( a_G ), b( a_B ), a( a_A )
		{
		}

		//! Return True if 2 Colours are identical
		bool operator ==( const Colour & o ) const
		{
			return r == o.r && g == o.g && b == o.b && a == o.a;
		}
		//! Return True if 2 Colours are different
		bool operator !=( const Colour & o ) const
		{
			return !( *this == o );
		}
	};
}

#endif // bee_color_h


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
