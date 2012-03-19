#ifndef _NAPALM_BUFFERSTORE__H_
#define _NAPALM_BUFFERSTORE__H_

#include "Object.h"


namespace napalm {

	// fwd decls
	class BufferStoreHolder;
	class Dispatcher;

	/*
	 * @class BufferStore
	 * @brief
	 * A BufferStore is a class which implements a storage system for a napalm buffer.
	 * Using buffer stores, you can change how and where a buffer stores its data.
	 * Some stores are 'directly' accessible, meaning that data can be read/written to as
	 * a contiguous block of data. Data like this is accessed via the virtual functions
	 * r(), rw() and w(). Other stores are more specialised and are not directly accessible.
	 *
	 * Here are some buffer store examples:
	 *
	 * *** A store to be used as the 'default' method of storing normal, cpu-based data.
	 * Such a store would be readable, read-writable, writable, resizable and not read-only.
	 * Napalm's default store implementation (BufferStoreCpu) satisfies this description.
	 *
	 * *** A store for exposing read-only data from an application which gives access to raw
	 * data arrays (such as maya). Such a store would be readable, and read-only, but would
	 * not be writable, read-writable or resizable.
	 *
	 * *** A store for moving data onto the GPU, for CUDA processing. Such a store would not
	 * be readable, read-writable or writable. It might be resizable (though probably not),
	 * but it would not be read-only (you probably want to be able to move cpu data into
	 * the buffer store).
	 *
	 * A Buffer-derived class uses a BufferStorage-derived object to store its data. In cases
	 * where client code tries to access the data via direct-access methods r(), rw() etc,
	 * but the data is not directly read/writable, the Buffer-derived class will silently
	 * swap the store contents into a cpu-based store.
	 */
	class BufferStore : public Object
	{
	public:

		BufferStore(bool r = false, bool rw = false, bool w = false,
			bool sz = false, bool cl = false, bool ro = false);

		virtual ~BufferStore(){}

		// todo collapse these flags into a bitset
		// the store contents can be read directly
		inline bool readable() const 		{ return m_readable; }

		// the store contents can be read/written directly
		inline bool readwritable() const 	{ return m_readwritable; }

		// the store contents can be (possibly destructively) written directly
		inline bool writable() const 		{ return m_writable; }

		// the store is resizable
		inline bool resizable() const 		{ return m_resizable; }

		// the store is clonable todo not sure about cloning + stores just yet...
		inline bool clonable() const 		{ return m_clonable; }

		// the store is readonly. This is not the same as 'readable and not writable' -
		// it implies that any call to copyFrom() will fail - the buffer cannot be written
		// to, directly nor indirectly.
		inline bool readonly() const		{ return m_readonly; }

		/*
		 * @brief isOrphan
		 * @returns True if this store is not owned by a buffer, false otherwise.
		 */
		bool isOrphan() const;

		/*
		 * @brief size
		 * @returns The number of elements in the store.
		 */
		virtual unsigned int size() const = 0;

		/*
		 * @brief resize
		 * Resize the data store.
		 * @param n New buffer size, in elements
		 * @param destructive If true, the contents of the buffer may be undefined afterwards
		 * @returns True if successful, false if the store is not resizable
		 */
		virtual bool resize(unsigned int n, bool destructive = false) { return false; }

		/*
		 * @brief clientSize
		 * @returns The number of elements in 'client' memory - that is, in CPU memory and
		 * under control of napalm.
		 */
		virtual unsigned int clientSize() const { return 0; }

		/*
		 * @brief shrink
		 * Discard any extra allocated memory that is no longer needed.
		 */
		virtual void shrink(){}

	protected:

		friend class BufferStoreHolder;
		friend class Dispatcher;

		virtual const std::type_info& elementType() const = 0;

		// set the parent store holder
		void setHolder(store_holder_ptr holder);

		friend class boost::serialization::access;
		template<class Archive> void serialize(Archive & ar, const unsigned int version);

	protected:

		bool m_readable;
		bool m_readwritable;
		bool m_writable;
		bool m_resizable;
		bool m_clonable;
		bool m_readonly;

		store_holder_wptr m_holder;
	};


///////////////////////// impl

// note: m_holder is deliberately not serialised
template<class Archive>
void BufferStore::serialize(Archive& ar, const unsigned int version)
{
	using boost::serialization::make_nvp;
	using boost::serialization::base_object;

	ar & make_nvp("base_class", base_object<Object>(*this));
	ar & make_nvp("readable", 		m_readable);
	ar & make_nvp("readwritable", 	m_readwritable);
	ar & make_nvp("writable", 		m_writable);
	ar & make_nvp("resizable", 		m_resizable);
	ar & make_nvp("clonable", 		m_clonable);
	ar & make_nvp("readonly", 		m_readonly);
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
