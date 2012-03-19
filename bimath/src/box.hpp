#ifndef _BIMATH_BOX__H_
#define _BIMATH_BOX__H_

#include "vec.hpp"
#include "traits.hpp"
#include <OpenEXR/ImathBox.h>
#include <iostream>


namespace bimath {

	template<typename T>
	struct imath_traits<Imath::Box<T> > {
		typedef T value_type;
		typedef typename imath_traits<T>::scalar_type scalar_type;
	};

} // bimath


namespace Imath {

	// missing from imath
	template<class T>
	std::ostream& operator <<(std::ostream& s, const Box<T>& b) {
		return s << '(' << b.min << ' ' << b.max << ')';
	}

}


namespace boost { namespace serialization {


	template<class Archive, class T>
	void serialize(Archive& ar, Imath::Box<T>& b, const unsigned int version)
	{
		using boost::serialization::make_nvp;
		ar & make_nvp("min", b.min);
		ar & make_nvp("max", b.max);
	}

	template<class T>
	struct is_bitwise_serializable<Imath::Box<T> > : public is_bitwise_serializable<T> {};

} }

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
