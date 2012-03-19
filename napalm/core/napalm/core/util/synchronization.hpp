#ifndef _NAPALM_UTIL_SYNCHRONIZATION_HPP_
#define _NAPALM_UTIL_SYNCHRONIZATION_HPP_

/*
 * @brief Generic interface to synchronization primitives. Actual implementation varies
 * depending on the value of NAPALM_SYNC_MODE.
 */

#ifndef NAPALM_SYNC_MODE

namespace napalm { namespace util {

	template<typename T>
	struct null_lock { null_lock(T){} };

	// a mutex
	typedef int							mutex;
	typedef null_lock<mutex> 			scoped_lock;

	// a read/write mutex
	typedef int							rw_mutex;
	typedef null_lock<rw_mutex> 		read_lock;
	typedef null_lock<rw_mutex> 		write_lock;

	// a mutex intended to be locked for extremely short time periods
	typedef int							spin_mutex;
	typedef null_lock<spin_mutex> 		spin_scoped_lock;

	// a read/write mutex intended to be locked for extremely short time periods
	typedef int							spin_rw_mutex;
	typedef null_lock<spin_rw_mutex> 	spin_read_lock;
	typedef null_lock<spin_rw_mutex> 	spin_write_lock;

} }

#elif NAPALM_SYNC_MODE == BOOST

#include <boost/thread/mutex.hpp>
#include <boost/thread/thread.hpp>

namespace napalm { namespace util {

	typedef boost::mutex						mutex;
	typedef boost::unique_lock<mutex> 			scoped_lock;

	typedef boost::shared_mutex					rw_mutex;
	typedef boost::shared_lock<rw_mutex> 		read_lock;
	typedef boost::unique_lock<rw_mutex> 		write_lock;

	typedef boost::mutex						spin_mutex;
	typedef boost::unique_lock<spin_mutex> 		spin_scoped_lock;

	typedef boost::shared_mutex					spin_rw_mutex;
	typedef boost::shared_lock<spin_rw_mutex> 	spin_read_lock;
	typedef boost::unique_lock<spin_rw_mutex> 	spin_write_lock;

} }

#elif NAPALM_SYNC_MODE == TBB
#error "Tbb synchronization not yet supported"
#else
#error "Unknown napalm synchronization mode requested: " #NAPALM_SYNC_MODE
#endif

#endif








