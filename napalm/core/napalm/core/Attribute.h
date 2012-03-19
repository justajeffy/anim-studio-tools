#ifndef _NAPALM_ATTRIBUTE__H_
#define _NAPALM_ATTRIBUTE__H_

#include "Object.h"


namespace napalm {


	/*
	 * @class Attribute
	 * @brief
	 * Virtual base class for all napalm attributes. An attribute is a single value,
	 * such as an integer or V3f.
	 */
	class Attribute : public Object
	{
	public:
		virtual ~Attribute(){}

		virtual const std::type_info& type() const = 0;

		bool operator==(const Attribute& a_Other) const;
		bool operator!=(const Attribute& a_Other) const;
		bool operator<(const Attribute & a_Other) const;

	protected:
		Attribute(){}

		friend class boost::serialization::access;
		template<class Archive> void serialize(Archive & ar, const unsigned int version);
	};


///////////////////////// impl

template<class Archive>
void Attribute::serialize(Archive& ar, const unsigned int version)
{
	using boost::serialization::make_nvp;
	using boost::serialization::base_object;
	ar & make_nvp("base_class", base_object<Object>(*this));
}


} // ns


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
