#ifndef _H5CPP_DATATYPE__H_
#define _H5CPP_DATATYPE__H_

/*
 * @file datatype.hpp
 * As well as providing wrapped API calls for H5Txxx() functions, this file also implements
 * a system for registering compound datatypes. This example illustrates:
 *
 * @code
 * struct foo {
 * 		int m_a;
 * 		char[10] m_b;
 * };
 *
 * compound_datatype_base_ptr createCompoundType_foo()
 * {
 *		typedef compound_datatype<foo> cdtype;
 *		typedef boost::shared_ptr<cdtype> cdptr;
 *
 *		cdptr cd(new cdtype());
 *		H5CPP_INSERT_COMPOUND_SUBTYPE(cd, m_a, "a");
 *		H5CPP_INSERT_COMPOUND_SUBTYPE(cd, m_b, "b");
 *		return cd;
 * }
 *
 * int main()
 * {
 *		datatype::register_compound_type<foo>(createCompoundType_foo);
 *
 *		foo f;
 *		attrib::create(some_group_id, "myfoo", f);
 * }
 * @endcode
 */

#include "hid_wrap.hpp"
#include "hdf5_string.hpp"
#include "util/type_info.hpp"
#include <boost/mpl/or.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_same.hpp>
#include <iostream>
#include <list>
#include <map>


namespace h5cpp {


	/*
	 * @class compound_datatype_base
	 * @brief Base class for compound_datatype.
	 */
	class compound_datatype_base
	{
	public:
		virtual ~compound_datatype_base();
		inline shared_hid_datatype shared_hid() const { return m_dtypes.front(); }
		inline const hid_datatype& hid() const { return m_dtypes.front().hid(); }

		template<typename T>
		void insert(std::size_t varoffset, const std::string& name);

	protected:
		compound_datatype_base(std::size_t size);
		compound_datatype_base(const compound_datatype_base& rhs);

	protected:
		std::list<shared_hid_datatype> m_dtypes;
	};

	typedef boost::shared_ptr<compound_datatype_base> compound_datatype_base_ptr;
	typedef compound_datatype_base_ptr(*compound_creator_fn)();


	/*
	 * @class compound_datatype
	 * @brief Creates an hdf5 compound datatype for the given type T.
	 */
	template<typename T>
	class compound_datatype : public compound_datatype_base
	{
	public:
		compound_datatype();
	};


	/*
	 * Helper macros for inserting types into compound datatype.
	 */
	#define H5CPP_INSERT_COMPOUND_SUBTYPE(CompoundObjPtr, MemberVar, MemberLabel) {							\
		char* objPtr;																						\
		std::size_t varoffset = (char*)(&(detail::createT(*CompoundObjPtr, &objPtr)->MemberVar)) - objPtr;	\
		detail::compound_datatype_insert(*CompoundObjPtr, varoffset, MemberLabel, 							\
			detail::createT(*CompoundObjPtr)->MemberVar);													\
	}


namespace datatype {

	/*
	 * @brief get_type
	 * Given a type T, return the hdf5 datatype registered for T.
	 * Supported native types (int, float etc) are listed in inc/native_types.inc.
	 * Compound types must be registered via register_compound_type() before calling get_type.
	 * The 'vl_string' and 'char*' types are registered as an hdf5 variable-length string.
	 * The 'char[N]' type is registered as an hdf5 fixed-length string of length N.
	 * The 'fl_string' type is not supported, since an instance of T is necessary to determine
	 * the string length (use the second form of get_type instead).
	 * Other string-related types (char*, char[], std::string) are not supported as they
	 * introduce confusing ambiguity.
	 */
	// todo deal with char*, char[N] then remove compound-type specialisations
	template<typename T>
	shared_hid_datatype get_type();

	#define _H5CPP_DEFN_INC(Type, H5Type) \
		template<> shared_hid_datatype get_type<Type>();
	#include "inc/native_types.inc"
	#undef _H5CPP_DEFN_INC

	template<> shared_hid_datatype get_type<vl_string>();


	/*
	 * @brief get_type
	 * Identical to the first form of get_type, except that it takes a reference to an
	 * instance of T. This form will also accept 'fl_string' as type T.
	 */
	template<typename T>
	shared_hid_datatype get_type(const T& t) { return get_type<T>(); }

	template<> shared_hid_datatype get_type<fl_string>(const fl_string& t);


	/*
	 * @brief get_fl_string_type
	 * Return an hdf5 datatype for a fixed-length string of the given length.
	 * @param nchars The number of characters, including the null-terminator.
	 */
	shared_hid_datatype get_fl_string_type(std::size_t nchars);


	/*
	 * @brief register_compound_type
	 * Register a compound datatype for type T. After this call, you will be able to create,
	 * load and write other hdf5 objects (attributes etc) based on the type T.
	 * @note T must be default-constructable.
	 */
	template<typename T>
	void register_compound_type(compound_creator_fn fn);


	/*
	 * @class compound_datatype_manager
	 * @brief
	 */
	class compound_datatype_manager
	{
	public:

		template<typename T>
		static void register_datatype(compound_creator_fn fn);

		template<typename T>
		static shared_hid_datatype get_datatype();

	protected:

		typedef std::map<const std::type_info*, compound_creator_fn,
			util::type_info_compare> map_type;

		static map_type m_compound_creator_fns;
	};

} // datatype ns

///////////////////////// impl

namespace detail {

	template<typename T>
	struct _cdtype_inserter
	{
		static shared_hid_datatype insert(hid_t cdtype_id, std::size_t varoffset, const std::string& name)
		{
			shared_hid_datatype dtype = datatype::get_type<T>();
			H5CPP_ERR_ON_NEG(H5Tinsert(cdtype_id, name.c_str(), varoffset, dtype.id()));
			return dtype;
		}
	};

	template<>
	struct _cdtype_inserter<char*>
	{
		static shared_hid_datatype insert(hid_t cdtype_id, std::size_t varoffset, const std::string& name);
	};

	template<std::size_t N>
	struct _cdtype_inserter<char[N]>
	{
		static shared_hid_datatype insert(hid_t cdtype_id, std::size_t varoffset, const std::string& name)
		{
			shared_hid_datatype dtype = datatype::get_fl_string_type(N);
			H5CPP_ERR_ON_NEG(H5Tinsert(cdtype_id, name.c_str(), varoffset, dtype.id()));
			return dtype;
		}
	};

} // detail ns


template<typename T>
void compound_datatype_base::insert(std::size_t varoffset, const std::string& name)
{
	// fl_string and vl_string are not to be used in compound types
	BOOST_MPL_ASSERT_NOT((boost::is_same<T, fl_string>));
	BOOST_MPL_ASSERT_NOT((boost::is_same<T, vl_string>));

	shared_hid_datatype dtype = detail::_cdtype_inserter<T>::insert(hid().id(), varoffset, name);
	m_dtypes.push_back(dtype);
}


template<typename T>
compound_datatype<T>::compound_datatype()
:	compound_datatype_base(sizeof(T))
{
}


namespace datatype {

template<typename T>
void compound_datatype_manager::register_datatype(compound_creator_fn fn)
{
	assert(fn);

	const std::type_info* ti = &typeid(T);
	map_type::const_iterator it = m_compound_creator_fns.find(ti);
	if(it == m_compound_creator_fns.end())
	{
		m_compound_creator_fns.insert(map_type::value_type(ti, fn));
	}
	else if(it->second != fn)
	{
		std::cerr << "A compound datatype registration for the type '"
			<< util::get_type_name<T>() << "' already exists; second registration ignored."
			<< std::endl;
	}
}


template<typename T>
shared_hid_datatype compound_datatype_manager::get_datatype()
{
	const std::type_info* ti = &typeid(T);
	map_type::const_iterator it = m_compound_creator_fns.find(ti);
	if(it == m_compound_creator_fns.end())
		return shared_hid_datatype();
	else
	{
		compound_datatype_base_ptr cd = it->second();
		return shared_hid_datatype(cd);
	}
}


// note: this creates a new instance of compound datatypes each time it's called (if T is a
// compound type). See hid_wrap_datatype.h for more info.
template<typename T>
shared_hid_datatype get_type()
{
	shared_hid_datatype dtype = compound_datatype_manager::get_datatype<T>();
	if(!dtype)
		H5CPP_THROW("get_type: type '" << util::get_type_name<T>() << "' is not registered with h5cpp");

	return dtype;
}


template<typename T>
void register_compound_type(compound_creator_fn fn)
{
	compound_datatype_manager::register_datatype<T>(fn);
}

} // datatype ns


namespace detail
{

	template<typename T>
	boost::shared_ptr<T> createT(const compound_datatype<T>& cdtype, char** objPtr = NULL)
	{
		boost::shared_ptr<T> p(new T());
		if(objPtr)
			*objPtr = (char*)(p.get());
		return p;
	}


	template<typename CompoundType, typename MemberType>
	void compound_datatype_insert(CompoundType& cdtype, std::size_t varoffset,
		const char* memberLabel, const MemberType& var)
	{
		cdtype.template insert<MemberType>(varoffset, memberLabel);
	}

} // detail ns


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
