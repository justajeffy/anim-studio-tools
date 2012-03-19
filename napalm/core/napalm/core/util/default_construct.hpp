#ifndef _NAPALM_DEFAULT_CONSTRUCT__H_
#define _NAPALM_DEFAULT_CONSTRUCT__H_

#include <bimath/vec.hpp>


namespace napalm { namespace util {

	/*
	 * Some data types default-construct to uninitialized data. This can cause problems
	 * down the line (such as boost-xml serialization, which doesn't seem to like NaNs).
	 * This file provides initialized contruction of all base types used by napalm.
	 */

	template<typename T> struct default_construction
	{ inline static T value() { return T(); } };

	template<> struct default_construction<half>
	{ inline static half value() { return 0; } };

	template<typename T> struct default_construction<Imath::Vec2<T> >
	{ inline static Imath::Vec2<T> value() { return Imath::Vec2<T>(0,0); } };

	template<typename T> struct default_construction<Imath::Vec3<T> >
	{ inline static Imath::Vec3<T> value() { return Imath::Vec3<T>(0,0,0); } };

	template<typename T> struct default_construction<Imath::Vec4<T> >
	{ inline static Imath::Vec4<T> value() { return Imath::Vec4<T>(0,0,0,0); } };

} } // ns

#endif
