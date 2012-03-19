#include <boost/python.hpp>
#include "Object.h"
#include "util/to_string.hpp"


using namespace napalm;
namespace bp = boost::python;

namespace {

	bool _eq(const Object& a, const Object& b)
	{
		// equality test on napalm types is pointer equality - use the areEqual() function
		// to test for contents equality. boost.python appears to do an equality test on
		// the address of the objects' containing shared_ptr by default, hence this binding.
		return (&a == &b);
	}

	object_ptr _clone(object_ptr self)
	{
		return make_clone(self);
	}

	std::string _to_string( Object & a )
	{
		return util::to_string<Object>().value(a);
	}

	std::string _py_string( Object & a )
	{
		return util::to_string<Object>().value(a, util::PYTHON);
	}

	std::string _tup_string( Object & a )
	{
		return util::to_string<Object>().value(a, util::TUPLES);
	}
}

void _napalm_export_Object()
{
	bp::class_<Object, object_ptr>("Object", bp::no_init)
		.def("clone", _clone)
		.def("__eq__", _eq)
		.def("__str__", _to_string)
		.def("__repr__", _to_string)
		.def("pyStr", _py_string)
		.def("tupStr", _tup_string)
		;
}
