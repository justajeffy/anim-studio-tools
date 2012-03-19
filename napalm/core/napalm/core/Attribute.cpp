#include "Attribute.h"
#include "Dispatcher.h"

namespace napalm {

bool Attribute::operator==(const napalm::Attribute& a_Other) const
{
	return napalm::Dispatcher::instance().attribEqual(*this, a_Other);
}

bool Attribute::operator!=(const napalm::Attribute& a_Other) const
{
	return !napalm::Dispatcher::instance().attribEqual(*this, a_Other);
}

bool Attribute::operator<(const napalm::Attribute & a_Other) const
{
	return napalm::Dispatcher::instance().attribLessThan(*this, a_Other);
}

}


/*
bool napalm::Attribute::operator==(const napalm::Attribute & a_Other) const
{
	return napalm::Dispatcher::instance().CompareEq(this, &a_Other);
}

bool napalm::Attribute::operator!=(const napalm::Attribute & a_Other) const
{
	return !napalm::Dispatcher::instance().CompareEq(this, &a_Other);
}

bool napalm::Attribute::operator<(const napalm::Attribute & a_Other) const
{
	return napalm::Dispatcher::instance().CompareLess(this, &a_Other);
}
*/
