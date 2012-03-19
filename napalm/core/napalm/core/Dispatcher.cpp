#include "Dispatcher.h"
#include "BufferStore.h"
#include "util/type_info.hpp"


namespace napalm {


Dispatcher& Dispatcher::instance()
{
	static Dispatcher inst;
	return inst;
}


std::string Dispatcher::getTypeLabel(const std::type_info& ti) const
{
	map_type::const_iterator it = m_map.find(&ti);
	return (it == m_map.end())?
		util::get_type_name(ti.name()) : it->second.m_typeLabel;
}


c_store_ptr Dispatcher::getSaveableStore(c_store_ptr store) const
{
	const std::type_info& ti = store->elementType();
	base_map_type::const_iterator it = m_basemap.find(&ti);
	assert(it != m_basemap.end());
	return it->second.m_get_saveable_store_fn(store);
}


bool Dispatcher::attribLessThan(const Attribute& a, const Attribute& b) const
{
	const std::type_info* ta = &(a.type());
	const std::type_info* tb = &(b.type());

	if(ta != tb)
		return (m_CompareOrder.find(ta)->second < m_CompareOrder.find(tb)->second);
	else
		return m_lt_comparisons.find(ta)->second(&a, &b);
}


bool Dispatcher::attribEqual(const Attribute& a, const Attribute& b) const
{
	const std::type_info* ta = &(a.type());
	const std::type_info* tb = &(b.type());

	if(ta != tb)
		return false;

	less_than_func fn = m_lt_comparisons.find(ta)->second;
	return !fn(&a, &b) && !fn(&b, &a);
}


Dispatcher::Comparison Dispatcher::attribCompare(const Attribute& a, const Attribute& b) const
{
	const std::type_info* ta = &(a.type());
	const std::type_info* tb = &(b.type());

	if(ta != tb)
	{
		return (m_CompareOrder.find(ta)->second < m_CompareOrder.find(tb)->second)?
			LESS_THAN : GREATER_THAN;
	}

	less_than_func fn = m_lt_comparisons.find(ta)->second;

	if(fn(&a, &b))
		return LESS_THAN;
	else if(fn(&b, &a))
		return GREATER_THAN;

	return EQUAL_TO;
}

}





















