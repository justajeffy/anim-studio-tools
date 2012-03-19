#ifndef _NAPALM_BUFFER__H_
#define _NAPALM_BUFFER__H_

#include "Object.h"
#include "BufferStoreHolder.h"


namespace napalm {


	/*
	 * @class Buffer
	 * @brief
	 * Virtual base class for all napalm buffers.
	 */
	class Buffer : public Object
	{
	public:

		virtual ~Buffer(){}

		/*
		 * @brief size
		 * @returns The number of elements in the buffer.
		 * @note See BufferStore::size
		 */
		unsigned int size() const;

		/*
		 * @brief clientSize
		 * @returns The number of elements in 'client' memory - that is, in CPU memory and
		 * under control of napalm.
		 */
		unsigned int clientSize() const;

		/*
		 * @brief shrink
		 * Discard any extra allocated memory that is no longer needed.
		 */
		void shrink();

		/*
		 * @brief getAttribs
		 * Get the buffer's attribute table.
		 */
		attrib_table_ptr 	getAttribs() 		{ return m_attribTable; }
		c_attrib_table_ptr 	getAttribs() const	{ return m_attribTable; }

		/*
		 * @brief setAttribs
		 * Set the buffer's attribute table, replacing the old.
		 * @param newAttribs The new attribute table.
		 */
		void setAttribs(attrib_table_ptr newAttribs);

		/*
		 * @brief uniqueStore
		 * @returns True if this buffer's store is not being shared, false otherwise.
		 */
		bool uniqueStore() const;

		/*
		 * @brief storeUseCount
		 * @returns The number of buffers sharing this buffer's store.
		 */
		unsigned int storeUseCount() const;

		/*
		 * @brief hasSharedStore
		 * @returns True if this buffer and rhs share the same store, false otherwise. For
		 * example, a buffer and its clones will share the same store, assuming none have
		 * been written to.
		 */
		bool hasSharedStore(const Buffer& rhs) const;

		/*
		 * @brief resize
		 * Resize the buffer.
		 * @param n Number of elements.
		 * @param destructive If true, the contents of the buffer may be undefined afterwards.
		 */
		virtual void resize(unsigned int n, bool destructive = false) = 0;

		// todo make protected I think
		// todo dont think we need an impl here, change to =0
		virtual std::ostream& str(std::ostream& os, object_rawptr_set& printed, util::StringMode a_Type = util::DEFAULT) const;

		virtual std::ostream& dump(std::ostream& os, object_rawptr_set& printed) const;

		// buffer store direct access. Asking for a writable store may result in the
		// current store being replaced with a cloned store.
		// todo should this be here?
		// todo add nonconst ver which takes bool arg, but const ver doesn't
		//virtual store_ptr getStore(bool readOnly) const = 0;

		// replace the current store. If pullData is true then the current store's data
		// will be copied into the new store, if possible. Returns false if 'pullData' is
		// true, but the new store will not accept data writes.
		// todo should this be here?
		//virtual bool setStore(store_ptr store, bool pullData) = 0;


	protected:

		Buffer(store_ptr store);

		Buffer(){}

		friend class boost::serialization::access;
		template<class Archive> void save(Archive& ar, const unsigned int version) const;
		template<class Archive> void load(Archive& ar, const unsigned int version);
		BOOST_SERIALIZATION_SPLIT_MEMBER()

		store_ptr _store() const;

		void _setStore(store_ptr store) const;

	protected:

		mutable store_holder_ptr m_store_holder;

		attrib_table_ptr m_attribTable;
	};


///////////////////////// impl

template<class Archive>
void Buffer::save(Archive& ar, const unsigned int version) const
{
	using boost::serialization::make_nvp;
	using boost::serialization::base_object;

	ar & make_nvp("base_class", base_object<Object>(*this));
	ar & make_nvp("store_holder", m_store_holder);
	ar & make_nvp("attribs", m_attribTable);
}


template<class Archive>
void Buffer::load(Archive& ar, const unsigned int version)
{
	using boost::serialization::make_nvp;
	using boost::serialization::base_object;

	ar & make_nvp("base_class", base_object<Object>(*this));
	ar & make_nvp("store_holder", m_store_holder);
	ar & make_nvp("attribs", m_attribTable);

	// see BufferStoreHolder::load
	m_store_holder->reparentStore();
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
