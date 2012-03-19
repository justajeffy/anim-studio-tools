#ifndef _NAPALM_BUFFERFIXEDRANGEITERATOR__H_
#define _NAPALM_BUFFERFIXEDRANGEITERATOR__H_

#include "BufferFixedRange.h"

namespace napalm {


	// fwd decl
	template<typename T> class TypedBuffer;


	/*
	 * @class BufferFixedRangeIterator
	 * @brief
	 * An iterator which contains a BufferFixedRange. This class exists only for convenience,
	 * so that TypedBuffers can supply begin/end functions rather than forcing the user to
	 * create an intermediary BufferFixedRange directly. Really, all the work happens in
	 * an iterator returned from begin() - an end() iterator actually does nothing.
	 */
	template<typename Iterator>
	class BufferFixedRangeIterator
	{
	public:

		typedef BufferFixedRange<Iterator> buffer_fixed_range_type;

		typedef typename std::input_iterator_tag						iterator_category;
		typedef typename buffer_fixed_range_type::difference_type 		difference_type;
		typedef typename buffer_fixed_range_type::value_type 			value_type;
		typedef typename buffer_fixed_range_type::reference 			reference;
		typedef typename buffer_fixed_range_type::pointer 				pointer;
		typedef typename buffer_fixed_range_type::index_type 			index_type;

		BufferFixedRangeIterator(){}
		BufferFixedRangeIterator(const buffer_fixed_range_type& bfr);

		BufferFixedRangeIterator& operator++();		// preinc
		BufferFixedRangeIterator operator++(int);	// postinc
		bool operator==(const BufferFixedRangeIterator& rhs) const;
		bool operator!=(const BufferFixedRangeIterator& rhs) const { return !(*this == rhs); }
		reference operator*() { return *m_it; }

		void reset() { m_bfr.reset(); }

	protected:

		boost::shared_ptr<buffer_fixed_range_type> m_bfr;
		Iterator m_it;
	};


///////////////////////// impl

template<typename Iterator>
BufferFixedRangeIterator<Iterator>::BufferFixedRangeIterator(const buffer_fixed_range_type& bfr)
: 	m_bfr(new buffer_fixed_range_type(bfr)),
	m_it(m_bfr->begin())
{}


template<typename Iterator>
BufferFixedRangeIterator<Iterator>&
BufferFixedRangeIterator<Iterator>::operator++()
{
	assert(m_bfr);

	++m_it;
	if(m_it >= m_bfr->end())
		m_bfr.reset();

	return *this;
}


template<typename Iterator>
BufferFixedRangeIterator<Iterator>
BufferFixedRangeIterator<Iterator>::operator++(int)
{
	BufferFixedRangeIterator _this(*this);
	return ++(*this);
}


template<typename Iterator>
bool BufferFixedRangeIterator<Iterator>::operator==(const BufferFixedRangeIterator& rhs) const
{
	if(m_bfr)
	{
		return (rhs.m_bfr)?
			((m_bfr == rhs.m_bfr) && (m_it == rhs.m_it)) : false;
	}
	else
		return !rhs.m_bfr;
}


}

#endif

















