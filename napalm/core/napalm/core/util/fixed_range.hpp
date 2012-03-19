#ifndef _NAPALM_FIXED_RANGE__H_
#define _NAPALM_FIXED_RANGE__H_

#include <utility>
#include <iterator>
#include <algorithm>
#include <vector>
#include <boost/mpl/assert.hpp>
#include <boost/type_traits.hpp>
#include <cassert>


namespace napalm { namespace util {


	/*
	 * @class fixed_range
	 * @brief
	 * A fixed_range is a pair of (begin, end) iterators which present an std::vector-like
	 * interface (however, a fixed_range cannot be resized). Iterator types must be random access.
	 */
	template<typename Iterator>
	struct fixed_range : public std::pair<Iterator,Iterator>
	{
		typedef Iterator													iterator;
		typedef std::pair<Iterator,Iterator> 								pair_type;
		typedef typename std::iterator_traits<Iterator>::difference_type 	difference_type;
		typedef typename std::iterator_traits<Iterator>::value_type 		value_type;
		typedef typename std::iterator_traits<Iterator>::reference 			reference;
		typedef typename std::iterator_traits<Iterator>::pointer 			pointer;
		typedef typename boost::make_unsigned<difference_type>::type 		index_type;

		fixed_range(){}

		fixed_range(const Iterator& begin, const Iterator& end): pair_type(begin,end){}

		template<typename Iterator2>
		fixed_range(const std::pair<Iterator2, Iterator2>& p): pair_type(p){}

		virtual ~fixed_range(){}

		inline Iterator begin() const { return this->first; }
		inline Iterator end() const { return this->second; }

		inline index_type size() const { return std::distance(this->first, this->second); }

		inline bool empty() const { return (this->first == this->second); }

		inline reference front() const { return *(this->first); }

		inline reference back() const {
			Iterator it(this->second);
			return *(--it);
		}

		inline reference operator[](index_type n) const {
			assert(n < this->size());
			return this->first[n];
		}
	};


} }

#endif
















