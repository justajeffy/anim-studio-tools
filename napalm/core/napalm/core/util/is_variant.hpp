#ifndef _NAPALM_IS_VARIANT__H_
#define _NAPALM_IS_VARIANT__H_

#include <boost/variant.hpp>
#include <boost/mpl/bool.hpp>


namespace napalm { namespace util {

	/*
	 * Metafunction which evaluates to mpl::true_ if the given type is a variant
	 */

	template<typename T>
	struct is_variant : public boost::mpl::false_{};

	template <BOOST_VARIANT_ENUM_PARAMS(typename T)>
	struct is_variant<boost::variant<BOOST_VARIANT_ENUM_PARAMS(T)> > : boost::mpl::true_{};

} }

#endif
