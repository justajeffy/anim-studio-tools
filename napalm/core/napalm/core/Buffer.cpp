#include "Buffer.h"
#include "BufferStoreCpu.h"
#include "Attribute.h"
#include "Table.h"
#include <cxxabi.h>


namespace napalm {


Buffer::Buffer(store_ptr store)
:	m_store_holder(new BufferStoreHolder()),
	m_attribTable(new AttribTable())
{
	m_store_holder->setStore(store);
}



unsigned int Buffer::size() const
{
	return _store()->size();
}


unsigned int Buffer::clientSize() const
{
	return _store()->clientSize();
}


void Buffer::shrink()
{
	_store()->shrink();
}


void Buffer::setAttribs(attrib_table_ptr newAttribs)
{
	assert(newAttribs);
	m_attribTable = newAttribs;
}


// todo dont think we need this impl anymore
std::ostream& Buffer::str(std::ostream& os, object_rawptr_set& printed, util::StringMode a_Type) const
{
	const BufferStore* pstore = m_store_holder->m_store.get();

	if( a_Type == util::DEFAULT )
	{
		os 	<< '<' << Dispatcher::instance().getTypeLabel(typeid(*this))
			<< " @ " << this << " (" << Dispatcher::instance().getTypeLabel(typeid(*pstore))
			<< '[' << pstore->size() << "] @ " << pstore << ")>";

		if(!m_attribTable->empty())
		{
			os << " attribs:";
			m_attribTable->str(os, printed, a_Type);
		}
	}
	else
	{
		os << "( 'UNTYPED DATA, CANNOT PRINT', ";
		m_attribTable->str( os, printed, a_Type );
		os << " )";
	}

	return os;
}


std::ostream& Buffer::dump(std::ostream& os, object_rawptr_set& printed) const
{
	const BufferStore* pstore = m_store_holder->m_store.get();

	os 	<< '<' << Dispatcher::instance().getTypeLabel(typeid(*this))
		<< " at " << this << " (" << Dispatcher::instance().getTypeLabel(typeid(*pstore))
		<< '[' << pstore->size() << "] at " << pstore << ")>\n";

	if(!m_attribTable->empty())
		m_attribTable->dump2(os, printed, false);

	return os;
}


bool Buffer::uniqueStore() const
{
	return m_store_holder.unique();
}


unsigned int Buffer::storeUseCount() const
{
	return m_store_holder.use_count();
}


bool Buffer::hasSharedStore(const Buffer& rhs) const
{
	return (rhs.m_store_holder == m_store_holder);
}


store_ptr Buffer::_store() const
{
	return m_store_holder->m_store;
}


void Buffer::_setStore(store_ptr store) const
{
	if(!m_store_holder.unique())
		m_store_holder.reset(new BufferStoreHolder());

	m_store_holder->setStore(store);
}


}






