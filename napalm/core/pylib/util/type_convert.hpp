#ifndef _NAPALM_UTIL_TYPE_CONVERT__H_
#define _NAPALM_UTIL_TYPE_CONVERT__H_

#include <boost/numeric/conversion/cast.hpp>

/*
 * Napalm treats chars and unsigned chars as numbers, whereas python treats them as strings.
 * This is a unique case, and these traits are supplied so that we don't have to write an
 * entire specialized binding for char/uchar.
 *
 * todo unless this gets more general, I think change the name from type_convert, it's too
 * generic and doesn't really describe what's going on here
 */

namespace napalm { namespace util {

	template<typename T>
	struct type_converter
	{
		typedef T type;
		static inline const T& to_python(const T& value) { return value; }
		static inline const T& from_python(const T& value) { return value; }
	};

	template<>
	struct type_converter<char>
	{
		typedef int type;
		static inline type to_python(const char& value) { return static_cast<type>(value); }
		static inline char from_python(const type& value) { return boost::numeric_cast<char>(value); }
	};

	template<>
	struct type_converter<unsigned char>
	{
		typedef unsigned int type;
		static inline type to_python(const unsigned char& value) { return static_cast<type>(value); }
		static inline unsigned char from_python(const type& value) { return boost::numeric_cast<unsigned char>(value); }
	};

} }

#endif
