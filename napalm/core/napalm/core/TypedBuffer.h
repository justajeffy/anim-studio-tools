#ifndef _NAPALM_TYPEDBUFFER__H_
#define _NAPALM_TYPEDBUFFER__H_

#include "meta.hpp"
#include "Buffer.h"
#include "Table.h"
#include "BufferStoreCpu.h"
#include "BufferFixedRangeIterator.hpp"
#include "util/default_construct.hpp"
#include "util/safe_convert.hpp"


namespace napalm {

	/*
	 * @class TypedBuffer
	 * @brief
	 * Napalm buffer class templatised on element type.
	 */
	template<typename T>
	class TypedBuffer : public Buffer
	{
	public:

		typedef T										value_type;
		typedef BufferFixedRange<T*>					w_type;
		typedef BufferFixedRange<const T*>				r_type;
		typedef BufferFixedRangeIterator<T*>			iterator;
		typedef BufferFixedRangeIterator<const T*>		const_iterator;
		typedef TypedBufferStore<T>						typed_store_type;
		typedef boost::shared_ptr<typed_store_type>		typed_store_ptr;

		/*
		 * @brief Create and take ownership of the given data store.
		 * @note Will throw if store is already owned by another buffer. Buffers can only
		 * end up sharing the same store via cloning.
		 */
		TypedBuffer(typed_store_ptr store);

		/*
		 * @brief Create with a given size and fill value
		 * @param size Number of elements
		 * @param value Fill value
		 */
		// (aj) change size to std::szie_type? This will mean we can remove that enable_if down there I think...
		TypedBuffer(unsigned int size = 0, const T& value = util::default_construction<T>::value());

		/*
		 * @brief Create from an iterator range
		 * @param begin Start of range
		 * @param end End of range
		 */
		template<typename Iterator>
		TypedBuffer(Iterator begin, Iterator end,
			typename boost::enable_if<boost::mpl::or_<
				boost::is_pointer<Iterator>, detail::has_value_type<Iterator>
			> >::type* dummy = 0);

		/*
		 * @brief Create from another (possibly differently) typed buffer.
		 * @param rhs Buffer to copy from
		 * @note Data is copied from rhs - cloning does not occur, even if rhs has the
		 * same element type.
		 */
		template<typename U>
		TypedBuffer(boost::shared_ptr<const TypedBuffer<U> > rhs);

		template<typename U>
		TypedBuffer(boost::shared_ptr<TypedBuffer<U> > rhs);

		virtual ~TypedBuffer(){}

		virtual void resize(unsigned int n, bool destructive = false);

		virtual object_ptr clone(object_clone_map& cloned) const;

		// store access
		// todo put back
		//virtual store_ptr getStore(bool readOnly) const;
		//virtual bool setStore(store_ptr store, bool pullData);

		// todo move these funcs into base class...
		// return the size of the buffer, in elements
		//unsigned int size() const;
		// shrink the buffer
		//void shrink();
		// return number of elements resident in memory and under napalm's control
		//unsigned int clientSize() const;

		// read access
		r_type 			r() const;
		const_iterator	r_begin() const 	{ return const_iterator(r()); }
		const_iterator 	r_end() const		{ return const_iterator(); }

		// read/write access
		w_type 			rw();
		iterator 		rw_begin()			{ return iterator(rw()); }
		iterator 		rw_end()			{ return iterator(); }

		// write access (potentially destructive)
		w_type 			w();
		iterator		w_begin()			{ return iterator(w()); }
		iterator 		w_end()				{ return iterator(); }

		virtual std::ostream& str(std::ostream& os, object_rawptr_set& printed, util::StringMode a_Type = util::DEFAULT) const;

	protected:

		typedef BufferStoreCpu<T> 				cpu_store;
		typedef boost::shared_ptr<cpu_store>	cpu_store_ptr;

		inline typed_store_ptr typedStore() const {
			typed_store_ptr s = boost::dynamic_pointer_cast<typed_store_type>(_store());
			assert(s);
			return s;
		}

		template<typename U>
		void init(boost::shared_ptr<const TypedBuffer<U> > rhs);

		//store_ptr _getStore(bool readOnly);

		friend class boost::serialization::access;
		template<class Archive> void serialize(Archive & ar, const unsigned int version);
	};


///////////////////////// impl

template<typename T>
TypedBuffer<T>::TypedBuffer(typed_store_ptr store)
:	Buffer(store)
{}


template<typename T>
TypedBuffer<T>::TypedBuffer(unsigned int size, const T& value)
:	Buffer(cpu_store_ptr(new cpu_store(size, value)))
{}


template<typename T>
template<typename Iterator>
TypedBuffer<T>::TypedBuffer(Iterator begin, Iterator end,
	typename boost::enable_if<boost::mpl::or_<
		boost::is_pointer<Iterator>, detail::has_value_type<Iterator> > >::type*)
:	Buffer(cpu_store_ptr(new cpu_store(std::distance(begin,end))))
{
	cpu_store_ptr cpuStore = util::static_pointer_cast<cpu_store>(_store());
	w_type frDest = cpuStore->rwImpl();
	typename w_type::iterator itDest = frDest.begin();

	for(Iterator itSrc=begin; itSrc!=end; ++itSrc, ++itDest)
		*itDest = util::safe_convert<T>(*itSrc);
}


// note: copies rhs's data but guarantees that rhs won't be forced into client space
template<typename T>
template<typename U>
TypedBuffer<T>::TypedBuffer(boost::shared_ptr<const TypedBuffer<U> > rhs)
:	Buffer(cpu_store_ptr(new cpu_store(rhs->size())))
{
	init<U>(rhs);
}


template<typename T>
template<typename U>
TypedBuffer<T>::TypedBuffer(boost::shared_ptr<TypedBuffer<U> > rhs)
:	Buffer(cpu_store_ptr(new cpu_store(rhs->size())))
{
	init<U>(rhs);
}


template<typename T>
template<typename U>
void TypedBuffer<T>::init(boost::shared_ptr<const TypedBuffer<U> > rhs)
{
	typedef typename TypedBuffer<U>::r_type rhs_r_type;

	// read from a clone of rhs to prevent movement of rhs's store
	boost::shared_ptr<TypedBuffer<U> > rhs_clone = make_clone(rhs);
	rhs_r_type frSrc = rhs_clone->r();

	cpu_store_ptr store = util::static_pointer_cast<cpu_store>(_store());
	w_type frDest = store->rwImpl();

	typename rhs_r_type::iterator it = frSrc.begin();
	typename w_type::iterator it2 = frDest.begin();
	for(; it!=frSrc.end(); ++it, ++it2)
		*it2 = util::safe_convert<T>(*it);

	m_attribTable = rhs_clone->getAttribs();
}


template<typename T>
object_ptr TypedBuffer<T>::clone(object_clone_map& cloned) const
{
	assert(cloned.find(this) == cloned.end());

	TypedBuffer* pclone = new TypedBuffer();
	pclone->m_store_holder = m_store_holder;
	pclone->m_attribTable = make_clone(m_attribTable, &cloned);

	boost::shared_ptr<TypedBuffer> obj(pclone);
	cloned.insert(object_clone_map::value_type(this, obj));
	return obj;
}

/*
template<typename T>
store_ptr TypedBuffer<T>::getStore(bool readOnly) const
{
	return const_cast<TypedBuffer<T>*>(this)->_getStore(readOnly);
}

// todo bit messy, maybe able to remove this
template<typename T>
store_ptr TypedBuffer<T>::_getStore(bool readOnly)
{
	store_ptr store = _store();

	if(readOnly || m_store_holder.unique())
		return store;

	if(store->clonable())
		store = make_clone(store);
	else
	{
		// switch data to cpu store since current store can't clone
		cpu_store_ptr cpuStore(new cpu_store(0));
		store->copyTo(cpuStore);
		store = cpuStore;
	}

	_setStore(store);
	return store;
}


template<typename T>
bool TypedBuffer<T>::setStore(store_ptr newStore, bool pullData)
{
	assert(newStore);

	store_ptr store = _store();
	if(newStore == store)
		return true;

	if(pullData)
	{
		if(newStore->readonly())
			return false;

		// attempt to copy data directly from current store to the new store
		if(!BufferStore::copy(store, newStore))
		{
			// can't direct copy, so create a temp cpuStore as a go-between
			cpu_store_ptr cpuStore(new cpu_store(0));
			if(!store->copyTo(cpuStore))
			{
				// todo these asserts will probably go, in favour of a buffer-mismatch
				// exception
				// this should not happen, there is no reason why a store should not
				// be able to copy its contents into a cpuStore
				assert(false);
				return false;
			}

			// clear current store early, to reduce max mamory footprint of this op
			store.reset();
			m_store_holder.reset();

			if(!newStore->copyFrom(cpuStore))
			{
				// this should not happen - newStore did not claim to be read-only,
				// so it should be able to copy data from a cpuStore
				assert(false);
				return false;
			}
		}
	}

	_setStore(newStore);
	return true;
}
*/


template<typename T>
void TypedBuffer<T>::resize(unsigned int n, bool destructive)
{
	typed_store_ptr store = typedStore();

	if(store->size() == n)
		return;

	if(m_store_holder.unique())
	{
		if(store->resize(n, destructive))
			return;
	}

	// switch data to cpu store
	if(destructive || (n == 0))
	{
		cpu_store_ptr cpuStore(new cpu_store(n));
		store = cpuStore;
	}
	else
	{
		cpu_store_ptr cpuStore(new cpu_store(0));
		store->copyTo(cpuStore);
		cpuStore->resize(n);
		store = cpuStore;
	}

	_setStore(store);
}


template<typename T>
typename TypedBuffer<T>::r_type TypedBuffer<T>::r() const
{
	typed_store_ptr store = typedStore();
	r_type fr = store->r();

	if(fr)
		return fr;
	else
	{
		// store doesn't support direct read - switch data to cpu store
		// todo fixme at the moment this creates a separate store for this single reader,
		// and not for all readers. Is this what I want?
		cpu_store_ptr cpuStore(new cpu_store(0));
		store->copyTo(cpuStore);
		_setStore(cpuStore);
		return cpuStore->rImpl();
	}
}


template<typename T>
typename TypedBuffer<T>::w_type TypedBuffer<T>::rw()
{
	typed_store_ptr store = typedStore();

	if(m_store_holder.unique())
	{
		w_type fr = store->rw();
		if(fr)
			// store is direct r/w-able and not shared, so return data directly
			return fr;
	}

	// switch data store to cpu
	cpu_store_ptr cpuStore(new cpu_store(0));
	store->copyTo(cpuStore);
	_setStore(cpuStore);
	return cpuStore->rwImpl();
}


template<typename T>
typename TypedBuffer<T>::w_type TypedBuffer<T>::w()
{
	typed_store_ptr store = typedStore();

	if(m_store_holder.unique())
	{
		w_type fr = store->w();
		if(fr)
			// store is direct writable and not shared, so return data directly
			return fr;
	}

	// switch to cpu store of same size (no need for data copy)
	cpu_store_ptr cpuStore(new cpu_store(store->size()));
	_setStore(cpuStore);
	return cpuStore->rwImpl();
}


template<typename T>
template<class Archive>
void TypedBuffer<T>::serialize(Archive& ar, const unsigned int version)
{
	using boost::serialization::make_nvp;
	using boost::serialization::base_object;
	ar & make_nvp("base_class", base_object<Buffer>(*this));
}

template<typename T>
std::ostream& TypedBuffer<T>::str(std::ostream& os, object_rawptr_set& printed, util::StringMode a_Type) const
{
	const BufferStore* pstore = _store().get();

	if( a_Type == util::DEFAULT )
	{
		os 	<< '<' << Dispatcher::instance().getTypeLabel(typeid(*this))
			<< " @ " << this << " (" << Dispatcher::instance().getTypeLabel(typeid(*pstore))
			<< '[' << pstore->size();

		os  << "] @ " << pstore << ")>";


		if(!m_attribTable->empty())
		{
			os << " attribs:";
			m_attribTable->str(os, printed, a_Type);
		}
	}
	else
	{
		os << "( [ ";
		r_type arr = r();
		for( unsigned int i = 0; i < arr.size(); ++i )
		{
			if( i != 0 )
				os << ", ";
			os << util::to_string<T>::value( arr[i], a_Type );
		}
		os << " ], ";
		m_attribTable->str( os, printed, a_Type );
		os << " )";
	}

	return os;
}


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
