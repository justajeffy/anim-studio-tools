#include "ValueWrapper.hpp"
#include "../Dispatcher.h"

namespace napalm {

bool detail::ValueWrapperLessThan(const Object& a, const Object& b)
{
	assert(false);
	return (&a < &b);
}

bool detail::ValueWrapperLessThan(const Attribute& a, const Attribute& b)
{
	return Dispatcher::instance().attribLessThan(a, b);
}

bool detail::ValueWrapperEqual(const Object& a, const Object& b)
{
	assert(false);
	return false;
}

bool detail::ValueWrapperEqual(const Attribute& a, const Attribute& b)
{
	return Dispatcher::instance().attribEqual(a, b);
}

}
