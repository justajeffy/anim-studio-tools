#ifndef _NAPALM_UTIL_ATOMIC__H_
#define _NAPALM_UTIL_ATOMIC__H_

/*
 * @brief Compiler-agnostic interface for atomic operations.
 * http://gcc.gnu.org/onlinedocs/gcc-4.3.2/gcc/Atomic-Builtins.html
 */

#include <boost/mpl/assert.hpp>


namespace napalm { namespace util {

	template<typename T>
	struct is_atomic : public boost::mpl::false_{};

#ifdef __GNUC__
	template<> struct is_atomic<int> 					: public boost::mpl::true_{};
	template<> struct is_atomic<unsigned int> 			: public boost::mpl::true_{};
	template<> struct is_atomic<long> 					: public boost::mpl::true_{};
	template<> struct is_atomic<unsigned long> 			: public boost::mpl::true_{};
	template<> struct is_atomic<long long> 				: public boost::mpl::true_{};
	template<> struct is_atomic<unsigned long long> 	: public boost::mpl::true_{};
#endif

	template<typename T>
	T add_and_fetch(T& t, T v)
	{
		BOOST_MPL_ASSERT((is_atomic<T>));
#ifdef __GNUC__
		return __sync_add_and_fetch(&t, v);
#endif
	}

} }

#endif







