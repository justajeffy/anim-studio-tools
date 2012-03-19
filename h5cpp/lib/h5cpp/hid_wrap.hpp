#ifndef _H5CPP_HID_WRAP__H_
#define _H5CPP_HID_WRAP__H_

#include <boost/shared_ptr.hpp>
#include "hid_wrap_datatype.h"
#include "hid_types.hpp"
#include "close.hpp"

// TODO rename all this wrap stuff to shared_hid

namespace h5cpp
{

	/*
	 * @class hid_wrap
	 * @brief A class which wraps an hid_xxx instance inside a shared reference. On
	 * destruction, the contained handle is closed.
	 */
	template<typename HID>
	class hid_wrap
	{
	public:

		typedef HID value_type;

		explicit hid_wrap(const HID& hid = HID()):m_hid_holder(new hid_holder(hid)){}
		hid_wrap(const hid_wrap& rhs):m_hid_holder(rhs.m_hid_holder){}
		inline const HID& hid() const { return m_hid_holder->m_hid; }
		inline hid_t id() const { return m_hid_holder->m_hid.id(); }
		inline operator bool() const { return bool(m_hid_holder->m_hid); }
		void reset(const HID& hid = HID()) { m_hid_holder.reset(new hid_holder(hid)); }

	protected:

		struct hid_holder
		{
			hid_holder(const HID& hid):m_hid(hid){}
			~hid_holder() { close(m_hid); }
			HID m_hid;
		};

		boost::shared_ptr<hid_holder> m_hid_holder;
	};


	#define _H5CPP_DEFN_INC(Derived, Base, IType) typedef hid_wrap<Derived> shared_##Derived;
	#include "inc/_hid_types.inc"
	#undef _H5CPP_DEFN_INC

	typedef hid_wrap<hid_location> shared_hid_location;


	/*
	 * traits to associate hid_xxx type with shared_hid_xxx type
	 */
	template<typename HID>
	struct shared_hid_type {
		typedef hid_wrap<HID> type;
	};

	template<>
	struct shared_hid_type<hid_datatype> {
		typedef shared_hid_datatype type;
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
