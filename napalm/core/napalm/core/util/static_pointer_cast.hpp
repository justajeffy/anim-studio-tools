#ifndef _NAPALM_UTIL_STATIC_POINTER_CAST__H_
#define _NAPALM_UTIL_STATIC_POINTER_CAST__H_

#include <boost/shared_ptr.hpp>


/*
 * @brief Wrapper for boost::static_pointer_cast, which changes to a dynamic_pointer_case
 * with non-null assertion in debug mode.
 */

namespace napalm { namespace util {

#ifdef NDEBUG
using boost::static_pointer_cast;
#else
template<class T, class U>
boost::shared_ptr<T> static_pointer_cast(const boost::shared_ptr<U>& r)
{
	if(!r)
		return boost::shared_ptr<T>();

	boost::shared_ptr<T> t = boost::dynamic_pointer_cast<T>(r);
	assert(t);
	return t;
}
#endif

} }

#endif
