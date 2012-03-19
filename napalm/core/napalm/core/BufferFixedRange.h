#ifndef _NAPALM_BUFFERFIXEDRANGE__H_
#define _NAPALM_BUFFERFIXEDRANGE__H_

#include "util/fixed_range.hpp"
#include "typedefs.h"


namespace napalm {


	// fwd decls
	class BufferStore;
	template<typename T> class TypedBufferStore;


	/*
	 * @class BufferFixedRange
	 * @brief
	 * A fixed_range which holds an internal reference to the buffer store it is accessing.
	 * This reference is necessary because some buffer store implementations may wish to
	 * replace or delete a buffer store from a different thread. Consider a buffer store
	 * class which is managed from a singleton LRU cache - a store may be removed from
	 * memory at any time, which would leave a fixed_range pointing at deallocated memory.
	 */
	template<typename Iterator>
	class BufferFixedRange : public util::fixed_range<Iterator>
	{
	public:

		typedef util::fixed_range<Iterator>						fixed_range_type;
		typedef typename fixed_range_type::difference_type 		difference_type;
		typedef typename fixed_range_type::value_type 			value_type;
		typedef typename fixed_range_type::reference 			reference;
		typedef typename fixed_range_type::pointer 				pointer;
		typedef typename fixed_range_type::index_type 			index_type;

		typedef boost::shared_ptr<TypedBufferStore<value_type> > 		typed_store_ptr;
		typedef boost::shared_ptr<const TypedBufferStore<value_type> > 	c_typed_store_ptr;

		BufferFixedRange():fixed_range_type(Iterator(0), Iterator(0)){}

		BufferFixedRange(const Iterator& begin, const Iterator& end, c_typed_store_ptr store)
		: fixed_range_type(begin, end), m_store(store){}

		operator bool() const { return bool(m_store); }

	protected:

		c_typed_store_ptr m_store;
	};


}

#endif













/***
    Copyright 2008-2012 Dr D Studios Pty Limited (ACN 127 184 954) (Dr. D Studios)

    This file is part of anim-studio-tools.

    anim-studio-tools is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    anim-studio-tools is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with anim-studio-tools.  If not, see <http://www.gnu.org/licenses/>.
***/
