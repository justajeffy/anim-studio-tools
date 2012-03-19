#include "Object.h"
#include "Dispatcher.h"


namespace napalm {


std::ostream& Object::strPtr(std::ostream& os) const
{
	os << '<' << Dispatcher::instance().getTypeLabel(typeid(*this)) << " @ " << this << '>';
	return os;

}

std::ostream& Object::str(std::ostream& os, object_rawptr_set& printed, util::StringMode a_Type) const
{
	return strPtr(os);
}


std::ostream& Object::str(std::ostream& os, util::StringMode a_Type ) const
{
	object_rawptr_set printed;
	return str(os, printed, a_Type);
}


std::ostream& Object::dump(std::ostream& os, object_rawptr_set& printed) const
{
	return (str(os, printed) << '\n');
}


std::ostream& Object::dump(std::ostream& os) const
{
	object_rawptr_set printed;
	return dump(os, printed);
}


std::ostream& operator <<(std::ostream& os, const Object& o)
{
	object_rawptr_set s;
	return o.str(os, s);
}


}



