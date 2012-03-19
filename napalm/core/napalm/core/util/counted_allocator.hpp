#ifndef _NAPALM_UTIL_COUNTED_ALLOCATOR__H_
#define _NAPALM_UTIL_COUNTED_ALLOCATOR__H_

#include <vector>
#include "../system.h"


namespace napalm { namespace util {


	/*
	 * @class counted_allocator
	 * @brief
	 * An allocator for use with std containers such as std::vector, which keeps track of
	 * the total amount of memory allocated.
	 * T: The element type being allocated.
	 * U: A type containing a static function 'count_bytes(long)', which accepts a positive
	 * long integer for allocations, and a negative long integer for deallocations.
	 *
	 * @note Adapted from:
	 * http://www.velocityreviews.com/forums/t637645-wrapping-std-vector-to-track-memory-usage.html
	 */
	template<typename T>
	class counted_allocator : public std::allocator<T>
	{
	public:
	    counted_allocator() throw() { }

		counted_allocator(const counted_allocator& __a) throw()
	      : std::allocator<T>(__a) { }

	    template<typename _Tp1>
	    counted_allocator(const counted_allocator<_Tp1>&) throw() { }

	    template<typename _Tp1>
	        struct rebind
	        { typedef counted_allocator<_Tp1> other; };

		typedef std::allocator<T> base_type;

		typedef typename base_type::pointer pointer;
		typedef typename base_type::size_type size_type;

		pointer allocate(size_type n) {
			NapalmSystem::count_bytes(n * sizeof(T));
			return this->base_type::allocate(n);
		}

		pointer allocate(size_type n, void const* hint) {
			NapalmSystem::count_bytes(n * sizeof(T));
			return this->base_type::allocate(n, hint);
		}

		void deallocate(pointer p, size_type n) {
			NapalmSystem::count_bytes(-n * sizeof(T));
			this->base_type::deallocate(p, n);
		}
	};


} }

#endif




