#ifndef _NAPALM_META__H_
#define _NAPALM_META__H_

#include <vector>
#include <boost/mpl/contains.hpp>
#include <boost/mpl/not.hpp>
#include <boost/mpl/and.hpp>
#include <boost/mpl/has_xxx.hpp>
#include <boost/mpl/remove.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/type_traits/is_convertible.hpp>
#include "util/bimath.h"
#include "util/less_than.hpp"
#include "util/counted_allocator.hpp"
#include "system.h"


/*
 * Common metafunctions
 */

namespace napalm {

	namespace detail {
		BOOST_MPL_HAS_XXX_TRAIT_DEF(value_type);
		BOOST_MPL_HAS_XXX_TRAIT_DEF(iterator);
	}

	class AttributeList;


	/*
	 * If T is a napalm base type then inherits from true_, otherwise inherits from false_.
	 * A 'base' type is a type which all common napalm types (attribute, buffer etc) are
	 * instantiated on.
	 */
	template<typename T> struct is_napalm_base_type : public boost::mpl::false_{};

#define _NAPALM_TYPE_OP(T, Label) \
	template<> struct is_napalm_base_type<T> : public boost::mpl::true_{};
	#include "types/all.inc"
#undef _NAPALM_TYPE_OP


	/*
	 * @brief If T is a napalm attribute type then inherits from true_, otherwise inherits
	 * from false_.
	 */
	template<typename T> struct is_napalm_attrib_type : public is_napalm_base_type<T>{};
	template<> struct is_napalm_attrib_type<bool> : public boost::mpl::true_{};
	template<typename T> struct is_napalm_attrib_type< std::vector<T, util::counted_allocator<T> > > : public is_napalm_base_type<T>{};
	template<typename T> struct is_napalm_attrib_type< std::set<T, util::less<T>, util::counted_allocator<T> > > : public is_napalm_base_type<T>{};
	template<> struct is_napalm_attrib_type<AttributeList> : public boost::mpl::true_{};


	/*
	 * If T has a nested 'value_type' type and 'iterator' type then inherits from true_,
	 * otherwise inherits from false_.
	 */
	template<typename T>
	struct is_iterable_sequence : public boost::mpl::and_<
		detail::has_value_type<T>,
		detail::has_iterator<T>
	>{};

}


#endif
