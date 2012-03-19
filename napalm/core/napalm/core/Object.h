#ifndef _NAPALM_OBJECT__H_
#define _NAPALM_OBJECT__H_

#include <boost/serialization/nvp.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/split_member.hpp>

#include <boost/enable_shared_from_this.hpp>
#include <boost/type_traits.hpp>
#include <boost/mpl/assert.hpp>

#include <iostream>
#include <set>

#include "util/static_pointer_cast.hpp"
#include "util/to_string.hpp"
#include "typedefs.h"

// make sure system.h gets included. This forces the library to initialise, even if
// the napalm library is linked statically.
#include "system.h"


namespace napalm {

	/*
	 * @class Object
	 * @brief
	 * Virtual base class for all napalm core types
	 */
	class Object : public boost::enable_shared_from_this<Object>
	{
	public:

		virtual ~Object(){}

		// return an identical copy of this object, or an empty pointer if this object
		// is not clonable.
		virtual object_ptr clone(object_clone_map& cloned) const { return object_ptr(); }

		// print a string representation of the object
		virtual std::ostream& str(std::ostream& os, object_rawptr_set& printed, util::StringMode a_Type = util::DEFAULT) const;

		// helper version of str
		std::ostream& str(std::ostream& os = std::cout, util::StringMode a_Type = util::DEFAULT) const;

		// print a more detailed tabulated representation of the object
		virtual std::ostream& dump(std::ostream& os, object_rawptr_set& printed) const;

		// helper version of dump
		std::ostream& dump(std::ostream& os = std::cout) const;

		// print to stream in form <type @ address>
		std::ostream& strPtr(std::ostream& os) const;

		// string representation of a null object ptr
		static inline const char* nullRepr() { return "<None>"; }

	protected:

		Object(){}

		// todo isolation stuff

		friend std::ostream& operator <<(std::ostream& os, const Object& o);

		friend class boost::serialization::access;
		template<class Archive> void serialize(Archive & ar, const unsigned int version){}
	};


///////////////////////// impl

// helper function for safe object cloning
// NOTE: works with both const- and non-const-qualified T
template<typename T>
boost::shared_ptr<typename boost::remove_cv<T>::type>
make_clone(const boost::shared_ptr<T>& obj, object_clone_map* cloned = NULL)
{
	typedef typename boost::remove_cv<T>::type T_;
	BOOST_MPL_ASSERT((boost::is_base_of<Object, T_>));

	if(!obj)
		return boost::shared_ptr<T_>();

	object_ptr objClone;

	if(cloned)
	{
		object_clone_map::const_iterator it = cloned->find(obj.get());
		if(it == cloned->end())
		{
			objClone = obj->clone(*cloned);
			cloned->insert(object_clone_map::value_type(obj.get(), objClone));
		}
		else
			objClone = it->second;
	}
	else
	{
		object_clone_map cloned2;
		objClone = obj->clone(cloned2);
	}

	if(!objClone)
		return boost::shared_ptr<T_>();

	return util::static_pointer_cast<T_>(objClone);
}

namespace util {
	template<>
	struct to_string<Object>
	{
		static std::string value(const Object& t, StringMode a_Mode = DEFAULT)
		{
			std::ostringstream strm;
			t.str(strm, a_Mode);
			return strm.str();
		}
	};
}
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
