#ifndef _NAPALM_DISPATCHER__H_
#define _NAPALM_DISPATCHER__H_

#include <set>
#include <bimath/half.hpp>
#include <boost/cast.hpp>
#include <boost/type_traits/is_floating_point.hpp>
#include "typedefs.h"
#include "util/safe_convert.hpp"
#include "util/less_than.hpp"
#include "Attribute.h"

#include "List.h"

namespace napalm {

	/*
	 * @class Dispatcher
	 * @brief
	 * The dispatcher is a singleton class which dispatches a set of methods to an
	 * equivalent function registered for a specific type T. It provides runtime-type-based
	 * functionality where, for whatever reason, other methods (such as virtual functions)
	 * are not sufficient.
	 */
	class Dispatcher
	{
	public:

		typedef c_store_ptr (*get_saveable_store_fn)(c_store_ptr);

		struct base_entry
		{
			get_saveable_store_fn m_get_saveable_store_fn;
		};

		struct entry
		{
			std::string m_typeLabel;
		};

		enum Comparison
		{
			LESS_THAN = 0,
			GREATER_THAN,
			EQUAL_TO
		};

		static Dispatcher& instance();

		// (aj) roll AddMapRow into here, use traits to suppress vec/set for some types
		template<typename T>
		void setTypeEntry(const entry& e) {
			m_map.insert(map_type::value_type(&typeid(T), e));
		}

		template<typename T>
		void setBaseTypeEntry(const base_entry& e) {
			m_basemap.insert(base_map_type::value_type(&typeid(T), e));
		}

		/*
		 * @brief getTypeLabel
		 * Return the string label associated with the given type.
		 */
		std::string getTypeLabel(const std::type_info& ti) const;

		/*
		 * @brief getSaveableStore
		 * Given a store, return a temporary copy of the store, appropriate for serialization.
		 */
		c_store_ptr getSaveableStore(c_store_ptr store) const;

		/*
		 * @brief extractAttribValue
		 * Get the value of an attribute cast to type T, if possible.
		 * @returns True if the value was extracted, false if the value could not be
		 * converted to T.
		 */
		template<typename T>
		bool extractAttribValue(const Attribute& a, T& outValue) const;

		/*
		 * @brief attribLessThan
		 * @returns True if a < b, false otherwise. Value comparison does not occur across
		 * type boundaries - rather, all of one attribute type are considered less than all
		 * of another. The order of type comparisons is arbitrary, but consistent.
		 */
		bool attribLessThan(const Attribute& a, const Attribute& b) const;

		/*
		 * @brief attribEqual
		 * @returns True if a and b hold an equal value. If their types differ then this
		 * function always returns false.
		 */
		bool attribEqual(const Attribute& a, const Attribute& b) const;

		/*
		 * @brief attribCompare
		 * @returns The comparison of two attributes.
		 */
		Comparison attribCompare(const Attribute& a, const Attribute& b) const;

	protected:

		template<typename To, typename From>
		static bool extractAttribValueT(const Attribute* a, void* outValue);

		template<typename T>
		static bool attribLessThanT(const Attribute* a, const Attribute* b);




		//////////////////////////////////////////
	public:

		// (aj) todo have a func here that does all the base-type entry setup!!!!



		template<typename T>
		void AddMapRow( )
		{
#define _NAPALM_TYPE_OP(F, N) \
			AddConversionEntry<T,F>();
#include "types/all.inc"
#undef _NAPALM_TYPE_OP

			addLessThanComparison<T>();
		}

		template<typename T>
		void AddSelfOnlyMapRow()
		{
			AddConversionEntry<T, T>();
			addLessThanComparison<T>();
		}

		template<typename T, typename F>
		typename boost::enable_if<
		boost::mpl::and_<util::is_safe_convertible<T,F>, util::is_safe_convertible<F,T> >, void >::type
		/*void*/ AddConversionEntry()
		{
			if( boost::is_floating_point<T>::value || !boost::is_floating_point<F>::value )
			{
				m_conversions.insert(value_convert_map::value_type(
					type_info_pair(&typeid(T),&typeid(F)), &Dispatcher::extractAttribValueT<T,F>));
			}
		}

		template<typename T, typename F>
		typename boost::disable_if<
		boost::mpl::and_<util::is_safe_convertible<T,F>, util::is_safe_convertible<F,T> >, void >::type
		/*void*/ AddConversionEntry()
		{
		}

		template<typename T>
		void addLessThanComparison()
		{
			const std::type_info* ti = &(typeid(T));
			m_lt_comparisons.insert(less_than_map::value_type(ti, &Dispatcher::attribLessThanT<T>));
			m_CompareOrder.insert(compare_order_map::value_type(ti, m_CompareOrder.size()));
		}

	protected:

		Dispatcher(){}

	protected:

		typedef std::map<const std::type_info*, base_entry> 						base_map_type;
		typedef std::map<const std::type_info*, entry> 								map_type;
		typedef std::map<const std::type_info*, int> 								compare_order_map;

		typedef bool(*value_convert_func)(const Attribute*, void*);
		typedef std::pair<const std::type_info*, const std::type_info*> 			type_info_pair;
		typedef std::map<type_info_pair, value_convert_func>						value_convert_map;

		typedef bool(*less_than_func)(const Attribute*, const Attribute*);
		typedef std::map<const std::type_info*, less_than_func>						less_than_map;

		base_map_type m_basemap;
		map_type m_map;
		value_convert_map m_conversions;
		less_than_map m_lt_comparisons;
		compare_order_map m_CompareOrder;
	};


///////////////////////// impl

template<typename T>
bool Dispatcher::extractAttribValue(const Attribute& a, T& outValue) const
{
	value_convert_map::const_iterator it = m_conversions.find(
		std::make_pair(&typeid(T), &(a.type())));

	return (it == m_conversions.end())?
		false : it->second(&a, static_cast<void*>(&outValue));
}


template<typename To, typename From>
bool Dispatcher::extractAttribValueT(const Attribute* a, void* outValue)
{
	const From& val = boost::polymorphic_downcast<const TypedAttribute<From>*>(a)->value();
	*(static_cast<To*>(outValue)) = util::safe_convert<To, From>(val);
	return true;
}


template<typename T>
bool Dispatcher::attribLessThanT(const Attribute* a, const Attribute* b)
{
	const T& val_a = static_cast<const TypedAttribute<T>*>(a)->value();
	const T& val_b = static_cast<const TypedAttribute<T>*>(b)->value();
	return util::less_than(val_a, val_b);
}


// Unfortunately this has to be here to prevent the mutual references between AttributeList
// and the dispatcher from preventing compilation.

template<typename T>
bool AttributeList::extractEntry( unsigned int index, T& value )
{
	unsigned int nelems = m_contents.size();
	if(nelems == 0)
	{
		throw "Trying to get element of empty array...";
	}
	else
	{
		index = (index>=0)? index%nelems : ((-index/nelems)*nelems + nelems) + index;
	}

	boost::shared_ptr<Attribute> attrib =
		boost::dynamic_pointer_cast<Attribute>(m_contents[index]);
	if( !attrib )
	{
		return false;
	}

	return Dispatcher::instance().extractAttribValue(*attrib, value);
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
