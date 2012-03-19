#include "TypedAttribBind.hpp"

namespace napalm {


attrib_topython::map_type attrib_topython::s_map;


bp::object attrib_topython::convert(const object_ptr& o, bool wrap)
{
	attrib_ptr attrib = boost::dynamic_pointer_cast<Attribute>(o);
	return (attrib)? convert(attrib, wrap) : bp::object(o);
}

bp::object attrib_topython::convert(const attrib_ptr& a, bool wrap)
{
	const std::type_info* pti = &typeid(*a.get());
	map_type::iterator it = s_map.find(pti);
	assert(it != s_map.end());
	return (it->second)(a, wrap);
}


}
