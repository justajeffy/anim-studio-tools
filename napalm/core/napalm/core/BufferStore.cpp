#include "BufferStore.h"

namespace napalm {


BufferStore::BufferStore(bool r, bool rw, bool w, bool sz, bool cl, bool ro)
:	m_readable(r),
	m_readwritable(rw),
	m_writable(w),
	m_resizable(sz),
	m_clonable(cl),
	m_readonly(ro)
{}


void BufferStore::setHolder(store_holder_ptr holder)
{
	assert(holder);
	m_holder = holder;
}


bool BufferStore::isOrphan() const
{
	return m_holder.expired();
}


}






