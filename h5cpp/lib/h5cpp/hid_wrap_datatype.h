#ifndef _H5CPP_HID_WRAP_DATATYPE__H_
#define _H5CPP_HID_WRAP_DATATYPE__H_

#include <boost/shared_ptr.hpp>
#include "hid_types.hpp"
#include "close.hpp"

// TODO change the name of this file to shared_hid_datatype

namespace h5cpp
{

	// fwd decl
	class compound_datatype_base;
	namespace datatype { class compound_datatype_manager; }


	/*
	 * @class shared_hid_datatype
	 * @brief This class is analogous to the other shared_hid_xxx classes. However, a
	 * shared_hid_datatype may hold an instance of a compound datatype, and in this case
	 * the compound type requires that its subtypes remain valid during its lifetime.
	 *
	 * @note This approach would not be necessary if compound types were stored in a static
	 * map, however this causes problems on process exit - hdf5 is unloaded, and further
	 * calls to H5Xclose() fail, which h5cpp then throws as an exception. Hdf5 does not
	 * provide an exit hook, if it did then we could solve this problem.
	 *
	 * todo screw it maybe just ignore failed H5xclose() calls on compound subtypes?
	 */
	class shared_hid_datatype
	{
	public:

		typedef hid_datatype value_type;

		explicit shared_hid_datatype(const hid_datatype& hid = hid_datatype());
		shared_hid_datatype(const shared_hid_datatype& rhs);
		const hid_datatype& hid() const;
		hid_t id() const;
		operator bool() const;
		void reset(const hid_datatype& hid = hid_datatype());

	protected:

		shared_hid_datatype(boost::shared_ptr<compound_datatype_base> cdtype);
		friend class datatype::compound_datatype_manager;

	protected:

		struct hid_holder
		{
			hid_holder(const hid_datatype& hid):m_hid(hid){}
			~hid_holder() { close(m_hid); }
			hid_datatype m_hid;
		};

		boost::shared_ptr<hid_holder> m_hid_holder;
		boost::shared_ptr<compound_datatype_base> m_cdtype;
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
