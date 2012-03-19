#ifndef _H5CPP_HID_ADAPTOR__H_
#define _H5CPP_HID_ADAPTOR__H_

#include "hid_wrap.hpp"
#include "meta.hpp"
#include <boost/mpl/and.hpp>
#include <boost/mpl/contains.hpp>
#include <boost/utility/enable_if.hpp>


namespace h5cpp {

	namespace detail {

		template<typename HidVariant, typename HID>
		struct is_hid_variant_seq_type
		{
			typedef typename boost::mpl::contains<
				typename HidVariant::sequence_type, HID
			>::type type;
		};

		template<typename HidVariant, typename SharedHID>
		struct is_hid_variant_seq_type2
		{
			typedef typename boost::mpl::contains<
				typename HidVariant::sequence_type,
				typename SharedHID::value_type
			>::type type;
		};

	}


	/*
	 * @class hid_adaptor
	 * @brief h5cpp functions which wrap the hdf5 API (such as attrib::create etc) often
	 * take existing hdf5 handles as arguments. h5cpp lets the user pass either an hid_t,
	 * a strongly-typed hid_xxx, or a shared_hid_xxx instance for these arguments, by using
	 * this adaptor class.
	 * Implicit casting for hid_variant types is handled as well - for example, an hid_file
	 * or a shared_hid_group will both implicitly cast into an hid_adaptor<hid_location>.
	 *
	 * @note you should never need to use this class yourself - let the h5cpp API functions
	 * do the implicit type conversion for you.
	 */
	template<typename HID>
	class hid_adaptor
	{
	public:

		typedef typename shared_hid_type<HID>::type _shared_hid_type;

		hid_adaptor(hid_t id = -1):m_hid(id){}
		hid_adaptor(const HID& hid):m_hid(hid){}
		hid_adaptor(const _shared_hid_type& shared_hid):m_hid(shared_hid.hid()){}

		// shadows constructor hid_variant<Seq>(HID)
		template<typename HID_>
		hid_adaptor(const HID_& hid,
			typename boost::enable_if<boost::mpl::and_<
				has_sequence_type<HID>,
				detail::is_hid_variant_seq_type<HID,HID_>
			> >::type* dummy = 0)
		: m_hid(hid){}

		// analogous to above but accepts a shared_hid_xxx type
		template<typename SharedHID>
		hid_adaptor(const SharedHID& shared_hid,
			typename boost::enable_if<boost::mpl::and_<
				has_sequence_type<HID>,
				has_value_type<SharedHID>,
				detail::is_hid_variant_seq_type2<HID,SharedHID>
			> >::type* dummy = 0)
		: m_hid(shared_hid.hid()){}

		hid_t id() const { return m_hid.id(); }
		const HID& hid() const { return m_hid; }
		operator bool() const { return bool(m_hid); }

	protected:

		HID m_hid;
	};


	#define _H5CPP_DEFN_INC(Derived, Base, IType) typedef hid_adaptor<Derived> Derived##_adaptor;
	#include "inc/hid_types.inc"
	#undef _H5CPP_DEFN_INC

	typedef hid_adaptor<hid_object> hid_object_adaptor;

	typedef hid_adaptor<hid_location> hid_location_adaptor;

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
