#ifndef _H5CPP_HID_TYPES__H_
#define _H5CPP_HID_TYPES__H_

#include <boost/strong_typedef.hpp>
#include <boost/type_traits.hpp>
#include <boost/mpl/assert.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/contains.hpp>
#include <hdf5/hdf5.h>
#include "is_constant.h"
#include "error.hpp"


/*
 * The classes defined in this file provide strongly-typed hdf5 handles. They do not
 * attempt to manage shared hdf5 resources - for example, an hid_file will not try to
 * close the file on destruction - they only serve to provide type-safe wrappers for
 * hdf5's 'hid_t' type.
 */

namespace h5cpp
{


	/*
	 * base class for all hid_xxx types
	 */
	class hid_object
	{
	public:

		hid_object(const hid_t id = -1) { set(id); }
		hid_object(const hid_object& rhs) { this->m_id = rhs.m_id; }

		hid_object& operator=(const hid_object& rhs) { this->m_id = rhs.m_id; return *this; }
		hid_object& operator=(const hid_t id) { set(id); return *this; }

		bool operator==(const hid_object& rhs) const { return m_id == rhs.m_id; }

		inline hid_t id() const { return m_id; }
		inline operator bool() const { return (m_id >= 0); }

	protected:

		hid_object(const hid_t id, H5I_type_t t) { set(id, t); }

		inline void set(const hid_t id)
		{
			if((id != -1) && !is_constant(id) && !H5Iis_valid(id))
				H5CPP_THROW("bad hdf5 id: " << id);

			m_id = id;
		}

		inline void set(const hid_t id, H5I_type_t t)
		{
			if(id != -1)
			{
				if(is_constant(t, id))
					m_id = id;
				else
				{
					if(!H5Iis_valid(id))
						H5CPP_THROW("bad hdf5 id: " << id);

					H5I_type_t id_t = H5Iget_type(id);
					if(id_t != t)
						H5CPP_THROW("hid_t type mismatch - expected " << t << ", got " << id_t);
				}
			}
			m_id = id;
		}

	protected:
		hid_t m_id;
	};


	/*
	 * strongly-typed hid_t wrapper classes
	 */
	#define _H5CPP_DEFN_INC(Derived, Base, IType)										\
	class Derived : public Base															\
	{																					\
	public:																				\
		static const H5I_type_t s_type_i = IType;										\
		static const char* s_type_str;													\
		Derived(const hid_t id = -1): Base(id, IType){}									\
		Derived(const Derived& rhs): Base(rhs){}										\
		Derived& operator=(const Derived& rhs) { this->m_id = rhs.m_id; return *this; }	\
		bool operator==(const Derived& rhs) const { return m_id == rhs.m_id; }			\
	};

	#include "inc/hid_types.inc"
	#undef _H5CPP_DEFN_INC


	/*
	 * variant type
	 */
	template<typename Sequence>
	class hid_variant : public hid_object
	{
	public:

		typedef Sequence sequence_type;

		hid_variant(const hid_variant& rhs) {
			m_id = rhs.m_id;
			m_type = rhs.m_type;
		}

		template<typename HID>
		hid_variant(const HID& rhs)
		{
			BOOST_MPL_ASSERT((boost::mpl::contains<Sequence,HID>));
			m_id = rhs.id();
			m_type = HID::s_type_i;
		}

		hid_variant& operator=(const hid_variant& rhs) {
			m_id = rhs.m_id;
			m_type = rhs.m_type;
			return *this;
		}

		bool operator==(const hid_variant& rhs) const {
			return (m_id == rhs.m_id) && (m_type == rhs.m_type);
		}

		H5I_type_t type() const { return m_type; }

	protected:
		H5I_type_t m_type;
	};

	typedef hid_variant<boost::mpl::vector<hid_file, hid_group> > hid_location;


	/*
	 * safe casting. Eg:
	 * hid_location loc = ...;
	 * hid_file file = h5_dynamic_cast<hid_file>(loc);
	 */
	template<typename HID>
	inline HID h5_dynamic_cast(const hid_object& rhs)
	{
		return (rhs && (H5Iget_type(rhs.id()) == HID::s_type_i))? HID(rhs.id()) : HID();
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
