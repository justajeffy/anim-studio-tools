#ifndef _BIMATH_VEC__H_
#define _BIMATH_VEC__H_

#include <OpenEXR/ImathVec.h>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/is_bitwise_serializable.hpp>
#include <boost/type_traits/is_arithmetic.hpp>
#include "traits.hpp"


namespace bimath {

	template<typename T>
	struct imath_traits<Imath::Vec2<T> > {
		typedef T value_type;
		typedef T scalar_type;
	};

	template<typename T>
	struct imath_traits<Imath::Vec3<T> > {
		typedef T value_type;
		typedef T scalar_type;
	};

	template<typename T>
	struct imath_traits<Imath::Vec4<T> > {
		typedef T value_type;
		typedef T scalar_type;
	};

} // bimath


namespace boost { namespace serialization {


	// Vec2
	template<class Archive, class T>
	void serialize(Archive& ar, Imath::Vec2<T>& v, const unsigned int version)
	{
		using boost::serialization::make_nvp;
		ar & make_nvp("x", v.x);
		ar & make_nvp("y", v.y);
	}

	template<class T>
	struct is_bitwise_serializable<Imath::Vec2<T> > : public is_arithmetic<T> {};


	// Vec3
	template<class Archive, class T>
	void serialize(Archive& ar, Imath::Vec3<T>& v, const unsigned int version)
	{
		using boost::serialization::make_nvp;
		ar & make_nvp("x", v.x);
		ar & make_nvp("y", v.y);
		ar & make_nvp("z", v.z);
	}

	template<class T>
	struct is_bitwise_serializable<Imath::Vec3<T> > : public is_arithmetic<T> {};


	// Vec4
	template<class Archive, class T>
	void serialize(Archive& ar, Imath::Vec4<T>& v, const unsigned int version)
	{
		using boost::serialization::make_nvp;
		ar & make_nvp("x", v.x);
		ar & make_nvp("y", v.y);
		ar & make_nvp("z", v.z);
		ar & make_nvp("w", v.w);
	}

	template<class T>
	struct is_bitwise_serializable<Imath::Vec4<T> > : public is_arithmetic<T> {};


} } // ns

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
