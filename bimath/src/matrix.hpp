#ifndef _BIMATH_MATRIX__H_
#define _BIMATH_MATRIX__H_

#include <OpenEXR/ImathMatrix.h>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/is_bitwise_serializable.hpp>
#include <boost/type_traits/is_arithmetic.hpp>
#include "traits.hpp"


namespace bimath {

	template<typename T>
	struct imath_traits<Imath::Matrix33<T> > {
		typedef T value_type;
		typedef T scalar_type;
	};

	template<typename T>
	struct imath_traits<Imath::Matrix44<T> > {
		typedef T value_type;
		typedef T scalar_type;
	};

} // bimath


namespace boost { namespace serialization {


	// Matrix33
	template<class Archive, class T>
	void serialize(Archive& ar, Imath::Matrix33<T>& m, const unsigned int version)
	{
		using boost::serialization::make_nvp;
		ar & make_nvp("x", m.x);
	}

	template<class T>
	struct is_bitwise_serializable<Imath::Matrix33<T> > : public is_arithmetic<T> {};


	// Matrix44
	template<class Archive, class T>
	void serialize(Archive& ar, Imath::Matrix44<T>& m, const unsigned int version)
	{
		using boost::serialization::make_nvp;
		ar & make_nvp("x", m.x);
	}

	template<class T>
	struct is_bitwise_serializable<Imath::Matrix44<T> > : public is_arithmetic<T> {};


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
