#ifndef _NAPALM_TYPEDBUFFERSTORE__H_
#define _NAPALM_TYPEDBUFFERSTORE__H_

#include "BufferStore.h"
#include "BufferFixedRange.h"

namespace napalm {


	/*
	 * @class TypedBufferStore
	 * @brief Napalm buffer-store class templatised on element type.
	 */
	template<typename T>
	class TypedBufferStore : public BufferStore
	{
	public:

		typedef T 								value_type;
		typedef BufferFixedRange<T*>			w_type;
		typedef BufferFixedRange<const T*>		r_type;

		TypedBufferStore(bool r = false, bool rw = false, bool w = false,
			bool sz = false, bool cl = false, bool ro = false);

		virtual ~TypedBufferStore(){}

		/*
		 * @brief r
		 * Get read access to the store contents. Returns a null fixed range if this
		 * store is not readable.
		 */
		virtual r_type r() const { return r_type(); }

		/*
		 * @brief rw
		 * Get read+write access to the store contents. Returns a null fixed range if
		 * this store is not read-writable.
		 */
		virtual w_type rw() { return w_type(); }

		/*
		 * @brief w
		 * Get write access to the store contents. Note that this call is potentially
		 * destructive, ie the store's contents may be undefined. Returns a null fixed
		 * range if this store is not writable.
		 */
		virtual w_type w() { return w_type(); }

		/*
		 * @brief copyTo
		 * Copy the contents of this store into another.
		 * @returns true on success, false if the destination store could not be written to.
		 */
		virtual bool copyTo(boost::shared_ptr<TypedBufferStore> destStore) const = 0;

		/*
		 * @brief copyFrom
		 * Copy the contents of another store into this one.
		 * @returns true on success, false if the source store could not be read from.
		 */
		virtual bool copyFrom(boost::shared_ptr<const TypedBufferStore> srcStore) = 0;

		/*
		 * @brief copy
		 * Copy the contents of src into dest.
		 * @returns True on success, false if not possible.
		 */
		static bool copy(boost::shared_ptr<const TypedBufferStore> src,
			boost::shared_ptr<TypedBufferStore> dest);

	protected:

		virtual const std::type_info& elementType() const { return typeid(T); }

		friend class boost::serialization::access;
		template<class Archive> void serialize(Archive & ar, const unsigned int version);
	};


///////////////////////// impl

template<typename T>
TypedBufferStore<T>::TypedBufferStore(bool r, bool rw, bool w, bool sz, bool cl, bool ro)
:	BufferStore(r,rw,w,sz,cl,ro)
{}


template<typename T>
bool TypedBufferStore<T>::copy(boost::shared_ptr<const TypedBufferStore<T> > src,
	boost::shared_ptr<TypedBufferStore<T> > dest)
{
	assert(src && dest);

	if(src == dest)
		return true;

	return (src->copyTo(dest) || dest->copyFrom(src));
}


template<typename T>
template<class Archive>
void TypedBufferStore<T>::serialize(Archive& ar, const unsigned int version)
{
	using boost::serialization::make_nvp;
	using boost::serialization::base_object;
	ar & make_nvp("base_class", base_object<BufferStore>(*this));
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
