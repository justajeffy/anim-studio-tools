#ifndef _BIMATH_HALF__H_
#define _BIMATH_HALF__H_

#include <OpenEXR/half.h>
#include <OpenEXR/halfLimits.h>
#include <OpenEXR/ImathLimits.h>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/split_free.hpp>
#include <boost/serialization/is_bitwise_serializable.hpp>

inline bool operator < ( half a, half b)
{
	return static_cast<float>(a) < static_cast<float>(b);
}

BOOST_SERIALIZATION_SPLIT_FREE(half)

namespace boost { namespace serialization {

	template<class Archive>
	void save(Archive & ar, const half& h, unsigned int version)
	{
		using boost::serialization::make_nvp;
		unsigned short bits = h.bits();
		ar & make_nvp("bits", bits);
	}

	template<class Archive>
	void load(Archive & ar, half& h, unsigned int version)
	{
		using boost::serialization::make_nvp;
		unsigned short bits;
		ar & make_nvp("bits", bits);
		h.setBits(bits);
	}

	template<>
	struct is_bitwise_serializable<half> : public boost::mpl::true_ {};

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
