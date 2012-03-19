#ifndef _NAPALM_UTIL_SAFE_CONVERT__H_
#define _NAPALM_UTIL_SAFE_CONVERT__H_

#include <boost/numeric/conversion/cast.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/mpl/and.hpp>
#include <boost/mpl/or.hpp>
#include <boost/type_traits.hpp>
#include "bimath.h"


namespace napalm { namespace util {

	/*
	 * This file defines the metafunction 'is_safe_convertible' and the function
	 * 'safe_convert'. A 'safe' conversion is one which does a boost::numeric_cast where
	 * applicable, rather than a static_cast. Some extra conversions are registered here
	 * that boost::is_convertible doesn't know about.
	 */

	namespace detail {

		template<typename To, typename From>
		struct safe_converter : public boost::is_convertible<From,To>
		{
			static inline To convert(const From& src) {
				return boost::numeric_cast<To>(src);
			}
		};

		// scalar to imath-type
		template<typename To, typename From>
		struct scalar_imath_safe_converter
		: public boost::is_convertible<From, typename bimath::imath_traits<To>::value_type>
		{
			typedef typename bimath::imath_traits<To>::value_type T;
			static inline To convert(const From& src) {
				return To(safe_converter<T,From>::convert(src));
			}
		};

		template<typename T, typename From>
		struct safe_converter<Imath::Vec2<T>, From>
		: public scalar_imath_safe_converter<Imath::Vec2<T>, From>{};

		template<typename T, typename From>
		struct safe_converter<Imath::Vec3<T>, From>
		: public scalar_imath_safe_converter<Imath::Vec3<T>, From>{};

		template<typename T, typename From>
		struct safe_converter<Imath::Vec4<T>, From>
		: public scalar_imath_safe_converter<Imath::Vec4<T>, From>{};

		template<typename T, typename From>
		struct safe_converter<Imath::Matrix33<T>, From>
		: public scalar_imath_safe_converter<Imath::Matrix33<T>, From>{};

		template<typename T, typename From>
		struct safe_converter<Imath::Matrix44<T>, From>
		: public scalar_imath_safe_converter<Imath::Matrix44<T>, From>{};

		// vec
		#define CONV(x) safe_converter<T,F>::convert(x)

		template<typename T, typename F>
		struct safe_converter<Imath::Vec2<T>, Imath::Vec2<F> > : public boost::is_convertible<F,T>
		{
			typedef Imath::Vec2<T> to_type;
			typedef Imath::Vec2<F> from_type;
			static inline to_type convert(const from_type& src) {
				return to_type(CONV(src.x), CONV(src.y));
			}
		};

		template<typename T, typename F>
		struct safe_converter<Imath::Vec3<T>, Imath::Vec3<F> > : public boost::is_convertible<F,T>
		{
			typedef Imath::Vec3<T> to_type;
			typedef Imath::Vec3<F> from_type;
			static inline to_type convert(const from_type& src) {
				return to_type(CONV(src.x), CONV(src.y), CONV(src.z));
			}
		};

		template<typename T, typename F>
		struct safe_converter<Imath::Vec4<T>, Imath::Vec4<F> > : public boost::is_convertible<F,T>
		{
			typedef Imath::Vec4<T> to_type;
			typedef Imath::Vec4<F> from_type;
			static inline to_type convert(const from_type& src) {
				return to_type(CONV(src.x), CONV(src.y), CONV(src.z), CONV(src.w));
			}
		};

		// matrix
		template<typename T, typename F>
		struct safe_converter<Imath::Matrix33<T>, Imath::Matrix33<F> > : public boost::is_convertible<F,T>
		{
			typedef Imath::Matrix33<T> to_type;
			typedef Imath::Matrix33<F> from_type;
			static inline to_type convert(const from_type& src) {
				return to_type(
					CONV(src.x[0][0]), CONV(src.x[0][1]), CONV(src.x[0][2]),
					CONV(src.x[1][0]), CONV(src.x[1][1]), CONV(src.x[1][2]),
					CONV(src.x[2][0]), CONV(src.x[2][1]), CONV(src.x[2][2]));
			}
		};

		template<typename T, typename F>
		struct safe_converter<Imath::Matrix44<T>, Imath::Matrix44<F> > : public boost::is_convertible<F,T>
		{
			typedef Imath::Matrix44<T> to_type;
			typedef Imath::Matrix44<F> from_type;
			static inline to_type convert(const from_type& src) {
				return to_type(
					CONV(src.x[0][0]), CONV(src.x[0][1]), CONV(src.x[0][2]), CONV(src.x[0][3]),
					CONV(src.x[1][0]), CONV(src.x[1][1]), CONV(src.x[1][2]), CONV(src.x[1][3]),
					CONV(src.x[2][0]), CONV(src.x[2][1]), CONV(src.x[2][2]), CONV(src.x[2][3]),
					CONV(src.x[3][0]), CONV(src.x[3][1]), CONV(src.x[3][2]), CONV(src.x[3][3]));
			}
		};

		// box
		template<typename T, typename F>
		struct safe_converter<Imath::Box<T>, Imath::Box<F> > : public boost::is_convertible<F,T>
		{
			typedef Imath::Box<T> to_type;
			typedef Imath::Box<F> from_type;
			static inline to_type convert(const from_type& src) {
				return to_type(CONV(src.min), CONV(src.max));
			}
		};

		#undef CONV

	} // detail ns


	/*
	 * @brief safe_convert
	 * Convert from type From to type To, doing boost numeric casts wherever possible.
	 */
	template<typename To, typename From>
	inline To safe_convert(const From& src)
	{
		typedef detail::safe_converter<To,From> safe_converter_type;
		return safe_converter_type::convert(src);
	}


	/*
	 * @brief is_safe_convertible
	 * Like boost::is_convertible, but picks up the extra conversions that are
	 * specialised above.
	 */
	template<typename To, typename From>
	struct is_safe_convertible : public detail::safe_converter<To,From>{};

} }

#endif





















