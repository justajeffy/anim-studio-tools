#ifndef _NAPALM_UTIL_OPERATOR_DETECT__H_
#define _NAPALM_UTIL_OPERATOR_DETECT__H_

#include <boost/type_traits/remove_reference.hpp>
#include <boost/type_traits/remove_cv.hpp>
#include <boost/static_assert.hpp>
#include "bimath.h"

/*
 * @brief Metafunctions for detecting support of operators (eg <, *, + etc).
 * Taken from: http://www.martinecker.com/wiki/index.php?title=Detecting_the_Existence_of_Operators_at_Compile-Time
 */

namespace napalm { namespace util {

	namespace detail
	{
		// A tag type returned by an operator for any struct in this namespace
		// when T does not support the operator.
		struct tag {};

		// This type soaks up any implicit conversions and makes the operator
		// less preferred than any other such operator found via ADL.
		struct any
		{
			// Conversion constructor for any type.
			template <class T>
			any(T const&);
		};

		// Fallback operators for types T that don't support the operator.
		tag operator < (any const&, any const&);

		// Two overloads to distinguish whether T supports a certain operator expression.
		// The first overload returns a reference to a two-element character array and is chosen if
		// T does not support the expression, such as ==, whereas the second overload returns a char
		// directly and is chosen if T supports the expression. So using sizeof(check(<expression>))
		// returns 2 for the first overload and 1 for the second overload.
		typedef char yes;
		typedef char (&no)[2];

		no check(tag);

		template <class T>
		yes check(T const&);


		template <class T>
		struct has_less_than_impl
		{
			static typename boost::remove_cv<typename boost::remove_reference<T>::type>::type const& x;
			static const bool value = sizeof(check(x < x)) == sizeof(yes);
		};

		template<>
		struct has_less_than_impl<half> {
			static const bool value = true;
		};
	}

	template <class T>
	struct has_less_than : detail::has_less_than_impl<T> {};

} }

#endif



