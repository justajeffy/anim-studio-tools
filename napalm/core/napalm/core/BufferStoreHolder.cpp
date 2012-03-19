#include "BufferStoreHolder.h"
#include "BufferStoreCpu.h"
#include "Buffer.h"
#include "exceptions.h"


namespace napalm {


BufferStoreHolder::BufferStoreHolder()
{
}


// see BufferStoreHolder::load
void BufferStoreHolder::reparentStore()
{
	m_store->setHolder(shared_from_this());
}


store_ptr BufferStoreHolder::getStore()
{
	return m_store;
}


void BufferStoreHolder::setStore(store_ptr store)
{
	assert(store);

	if(store == m_store)
		return;

	if(!store->isOrphan())
	{
		NAPALM_THROW(NapalmError, "attempted to reparent non-orphan store "
			<< store.get() << " (" << Dispatcher::instance().getTypeLabel(typeid(*store.get())) << ")");
	}

	m_store = store;
	m_store->setHolder(shared_from_this());
}


}







