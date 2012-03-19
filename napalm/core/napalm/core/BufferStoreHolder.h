#ifndef _NAPALM_BUFFERSTOREHOLDER__H_
#define _NAPALM_BUFFERSTOREHOLDER__H_

#include "BufferStore.h"
#include "Dispatcher.h"


namespace napalm {


	// fwd decl
	class Buffer;

	/*
	 * @class BufferStoreHolder
	 */
	class BufferStoreHolder : public boost::enable_shared_from_this<BufferStoreHolder>
	{
	protected:

		friend class Buffer;

		BufferStoreHolder();

		void setStore(store_ptr store);

		store_ptr getStore();

		// see load() implementation
		void reparentStore();

		//store_ptr getSaveableStore() const;

		friend class boost::serialization::access;
		template<class Archive> void save(Archive& ar, const unsigned int version) const;
		template<class Archive> void load(Archive& ar, const unsigned int version);
		BOOST_SERIALIZATION_SPLIT_MEMBER()

	protected:

		store_ptr m_store;
	};


///////////////////////// impl

template<class Archive>
void BufferStoreHolder::save(Archive& ar, const unsigned int version) const
{
	using boost::serialization::make_nvp;

	c_store_ptr store = Dispatcher::instance().getSaveableStore(m_store);
	ar & make_nvp("store", store);
}


template<class Archive>
void BufferStoreHolder::load(Archive& ar, const unsigned int version)
{
	using boost::serialization::make_nvp;

	// note: m_store is only partially setup here - it still needs to have its parent holder
	// set to us. However this can't be done - boost is not placing our class instance
	// inside a shared_ptr before serialising, so a bad_weak_ptr exception would result.
	// We're relying on this being done in Buffer's serialization function.
	ar & make_nvp("store", m_store);
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
