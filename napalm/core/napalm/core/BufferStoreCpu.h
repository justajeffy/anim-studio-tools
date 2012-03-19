#ifndef _NAPALM_BUFFERSTORECPU__H_
#define _NAPALM_BUFFERSTORECPU__H_

#include <vector>
#include "TypedBufferStore.h"
#include "BufferFixedRange.h"
#include "delayed_vector.hpp"
#include "util/static_pointer_cast.hpp"
#include "io.h"


namespace napalm {


	// fwd decl
	template<typename T>
	class TypedBuffer;


	/*
	 * @class BufferStoreCpu
	 * @brief
	 * A buffer storage implementation based on std::vector. This class is used as the
	 * default method of buffer storage. Its data is direct-accessible for reads, writes
	 * and read/writes.
	 */
	template<typename T>
	class BufferStoreCpu : public TypedBufferStore<T>
	{
	public:

		typedef T										value_type;
		typedef TypedBufferStore<T>						base_class;
		typedef typename base_class::w_type				w_type;
		typedef typename base_class::r_type				r_type;

		BufferStoreCpu(unsigned int size = 0, const T& value = T());

		BufferStoreCpu(const BufferStoreCpu& rhs);

		virtual ~BufferStoreCpu(){}

		virtual unsigned int size() const { return m_data.size(); }
		virtual void shrink();
		virtual bool resize(unsigned int n, bool destructive = false);

		virtual r_type r() const;
		virtual w_type rw();
		virtual w_type w();

		virtual object_ptr clone(object_clone_map& cloned) const;
		virtual bool copyTo(boost::shared_ptr<base_class> destStore) const;
		virtual bool copyFrom(boost::shared_ptr<const base_class> srcStore);
		virtual unsigned int clientSize() const;

		// todo
		// static bool forceToClient(base_class_ptr store)

	protected:

		typedef typename delayed_vector<T>::vector_type 	vector_type;
		typedef boost::shared_ptr<base_class>				base_class_ptr;
		typedef boost::shared_ptr<const base_class>			c_base_class_ptr;

		r_type rImpl() const;
		w_type rwImpl();
		w_type wImpl();

		friend class TypedBuffer<T>;

		friend class boost::serialization::access;
		template<class Archive> void serialize(Archive& ar, const unsigned int version);

	protected:

		delayed_vector<T> m_data;
	};


	namespace detail {

		template<typename T>
		c_store_ptr getSaveableStore(c_store_ptr store);

	}


///////////////////////// impl

namespace detail {

	template<typename T>
	c_store_ptr getSaveableStore(c_store_ptr store)
	{
		typedef TypedBufferStore<T> 						typed_store_t;
		typedef BufferStoreCpu<T> 							typed_cpustore_t;
		typedef boost::shared_ptr<const typed_store_t> 		c_typed_store_ptr;
		typedef boost::shared_ptr<typed_cpustore_t> 		typed_cpustore_ptr;
		typedef boost::shared_ptr<const typed_cpustore_t> 	c_typed_cpustore_ptr;

		{
			// test for store that's already on cpu
			c_typed_cpustore_ptr cpuStore = boost::dynamic_pointer_cast<const typed_cpustore_t>(store);
			if(cpuStore)
				return cpuStore;
		}

		// create new cpu store and copy into it
		// todo when a BFR is incorporated here for threading reasons, remember to r() from a clone
		// rather than the original, so that typedstore's contents aren't forced clientside.
		// todo consider a BS::swap() fn to avoid the extra data copy?
		c_typed_store_ptr typedstore = util::static_pointer_cast<const typed_store_t>(store);
		typed_cpustore_ptr cpuStore(new typed_cpustore_t(0));
		typedstore->copyTo(cpuStore);

		return cpuStore;
	}

} // detail ns


template<typename T>
BufferStoreCpu<T>::BufferStoreCpu(unsigned int size, const T& value)
:	TypedBufferStore<T>(true, true, true, true, true, false),
	m_data(size, value)
{}


template<typename T>
BufferStoreCpu<T>::BufferStoreCpu(const BufferStoreCpu<T>& rhs)
:	TypedBufferStore<T>(true, true, true, true, true, false),
	m_data(rhs.m_data)
{}


template<typename T>
void BufferStoreCpu<T>::shrink()
{
	m_data.shrink();
}


template<typename T>
bool BufferStoreCpu<T>::resize(unsigned int n, bool destructive)
{
	m_data.resize(n, destructive);
	return true;
}


template<typename T>
typename BufferStoreCpu<T>::r_type BufferStoreCpu<T>::r() const
{
	return rImpl();
}


template<typename T>
typename BufferStoreCpu<T>::w_type BufferStoreCpu<T>::rw()
{
	return rwImpl();
}


template<typename T>
typename BufferStoreCpu<T>::w_type BufferStoreCpu<T>::w()
{
	return wImpl();
}


template<typename T>
typename BufferStoreCpu<T>::r_type BufferStoreCpu<T>::rImpl() const
{
	c_base_class_ptr pthis = util::static_pointer_cast<const base_class>(this->shared_from_this());

	const vector_type& vdata = m_data.data();
	return (vdata.empty())?
		r_type(NULL, NULL, pthis) : r_type(&vdata[0], &vdata[0]+vdata.size(), pthis);
}


template<typename T>
typename BufferStoreCpu<T>::w_type BufferStoreCpu<T>::rwImpl()
{
	c_base_class_ptr pthis = util::static_pointer_cast<const base_class>(this->shared_from_this());

	vector_type& vdata = m_data.data();
	return (vdata.empty())?
		w_type(NULL, NULL, pthis) : w_type(&vdata[0], &vdata[0]+vdata.size(), pthis);
}


template<typename T>
typename BufferStoreCpu<T>::w_type BufferStoreCpu<T>::wImpl()
{
	if(!m_data.isLoaded())
		m_data.resize(m_data.size(), true);
	return rwImpl();
}


template<typename T>
object_ptr BufferStoreCpu<T>::clone(object_clone_map& cloned) const
{
	assert(cloned.find(this) == cloned.end());
	object_ptr objClone(new BufferStoreCpu<T>(*this));
	cloned.insert(object_clone_map::value_type(this, objClone));

	return objClone;
}


template<typename T>
bool BufferStoreCpu<T>::copyTo(boost::shared_ptr<TypedBufferStore<T> > destStore) const
{
	assert(destStore);

	BufferStoreCpu<T>* pdest = dynamic_cast<BufferStoreCpu<T>*>(destStore.get());
	if(pdest)
	{
		// keeps non-resident data on disk
		pdest->m_data = m_data;
		return true;
	}

	if(!destStore->writable())
		return false;

	if(!destStore->resize(m_data.size()))
		return false;

	w_type frDest = destStore->w();
	assert(frDest.size() == m_data.size());

	if(m_data.isLoaded())
	{
		const vector_type& data = m_data.data();
		std::copy(data.begin(), data.end(), frDest.begin());
	}
	else
	{
		// our data is not resident - copy from a temp vector instead
		delayed_vector<T> tmpData(m_data);
		const vector_type& data = tmpData.data();
		std::copy(data.begin(), data.end(), frDest.begin());
	}

	return true;
}


template<typename T>
bool BufferStoreCpu<T>::copyFrom(boost::shared_ptr<const TypedBufferStore<T> > srcStore)
{
	assert(srcStore);

	// this stops srcStore's data from being loaded unnecessarily
	const BufferStoreCpu<T>* psrc = dynamic_cast<const BufferStoreCpu<T>*>(srcStore.get());
	if(psrc)
	{
		*this = *psrc;
		return true;
	}

	if(!srcStore->readable())
		return false;

	r_type frSrc = srcStore->r();

	m_data.resize(srcStore->size(), true);
	std::copy(frSrc.begin(), frSrc.end(), m_data.data().begin());

	return true;
}


template<typename T>
unsigned int BufferStoreCpu<T>::clientSize() const
{
	return m_data.clientSize();
}


template<typename T>
template<class Archive>
void BufferStoreCpu<T>::serialize(Archive& ar, const unsigned int version)
{
	using boost::serialization::make_nvp;
	using boost::serialization::base_object;

	ar & make_nvp("base_class", base_object<base_class>(*this));
	ar & make_nvp("data", m_data);
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
