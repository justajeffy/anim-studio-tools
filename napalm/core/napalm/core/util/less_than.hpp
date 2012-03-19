#ifndef _NAPALM_UTIL_LESS_THAN__H_
#define _NAPALM_UTIL_LESS_THAN__H_

#include <vector>
#include <set>
#include "bimath.h"
#include <boost/utility/enable_if.hpp>
#include "operator_detect.hpp"

/*
 * @brief Functions for performing less-than comparisons on various types. These are
 * supplied for two reasons:
 * (1) Some types (such as Imath vectors, matrices) do not provide a less-than operator.
 * (2) stl containers such as std::set compare container length first, and then contents.
 * We desire different behaviour - content first, and then length.
 */

namespace napalm { namespace util {

	template<typename T>
	inline typename boost::enable_if<has_less_than<T>, bool>::type
	/*bool*/ less_than(const T& a, const T& b)
	{
		return a < b;
	}

	template<typename T>
	bool less_than(const Imath::Vec2<T>& a, const Imath::Vec2<T> &b)
	{
		if( a[0] > b[0] )
			return false;
		return a[0] < b[0] || a[1] < b[1];
	}

	template<typename T>
	bool less_than(const Imath::Vec3<T>& a, const Imath::Vec3<T>& b)
	{
		for( int i = 0; i < 3; ++i )
		{
			if( a[i] < b[i] )
				return true;
			if( a[i] > b[i] )
				return false;
		}
		return false;
	}

	template<typename T>
	bool less_than(const Imath::Vec4<T>& a, const Imath::Vec4<T>& b)
	{
		for( int i = 0; i < 4; ++i )
		{
			if( a[i] < b[i] )
				return true;
			if( a[i] > b[i] )
				return false;
		}
		return false;
	}

	template<typename T>
	bool less_than(const Imath::Box<T>& a, const Imath::Box<T>& b)
	{
		if( less_than( b.min, a.min ) )
			return false;

		return less_than(a.min, b.min) || less_than(a.max, b.max);
	}

	template<typename T>
	bool less_than(const Imath::Matrix33<T>& a, const Imath::Matrix33<T>& b)
	{
		for( int i = 0; i < 3; ++i )
		{
			for( int j = 0; j < 3; ++j )
			{
				if( a[i][j] < b[i][j] )
					return true;
				if( a[i][j] > b[i][j] )
					return false;
			}
		}
		return false;
	}

	template<typename T>
	bool less_than(const Imath::Matrix44<T>& a, const Imath::Matrix44<T>& b)
	{
		for( int i = 0; i < 4; ++i )
		{
			for( int j = 0; j < 4; ++j )
			{
				if( a[i][j] < b[i][j] )
					return true;
				if( a[i][j] > b[i][j] )
					return false;
			}
		}
		return false;
	}

	template<typename T, typename Q>
	bool less_than(const std::vector<T,Q>& a, const std::vector<T,Q>& b)
	{
		typename std::vector<T,Q>::const_iterator it = a.begin();
		typename std::vector<T,Q>::const_iterator it2 = b.begin();
		for( ; it != a.end() && it2 != b.end(); ++it, ++it2 )
		{
			if(less_than(*it, *it2))
				return true;
			if(less_than(*it2, *it))
				return false;
		}

		return it2 != b.end();
	}

	template<typename T, typename Q, typename R>
	bool less_than(const std::set<T,Q,R>& a, const std::set<T,Q,R>& b)
	{
		if(a.size() < b.size())
			return true;
		if(b.size() < a.size())
			return false;

		typename std::set<T,Q,R>::const_iterator it = a.begin();
		typename std::set<T,Q,R>::const_iterator it2 = b.begin();

		for(; it!=a.end(); ++it, ++it2 )
		{
			if(less_than( *it, *it2))
				return true;
			if(less_than( *it2, *it))
				return false;
		}

		return false;
	}


	/*
	 * @class less
	 * @brief Less-than comparator for use in stl containers etc.
	 */
	template< typename T >
	struct less
	{
		inline bool operator()(const T& a, const T& b) {
			return less_than(a, b);
		}
	};

} } // ns


#endif
