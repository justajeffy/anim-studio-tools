#ifndef _BIMATH_TRAITS__H_
#define _BIMATH_TRAITS__H_

namespace bimath {

	/*
	 * @brief imath_traits
	 * Imath doesn't provide standard nested typedefs across its types - this traits
	 * class provides them instead.
	 */
	template<typename T>
	struct imath_traits
	{
		// The type the imath class is templatised on, eg float for v3f.
		typedef void value_type;

		// the scalar type the imath class is based on, may differ from value_type. Eg,
		// Box<V3f>'s scalar_type is float, but its value_type is v3f.
		typedef void scalar_type;
	};

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
