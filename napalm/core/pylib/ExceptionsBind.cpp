#include "ExceptionsBind.hpp"

using namespace napalm;

namespace {

	PyObject* g_excType(NULL);

	static void translate(const NapalmError& e)
	{
		bp::object pythonExceptionInstance(e);
		PyErr_SetObject(g_excType, pythonExceptionInstance.ptr());
	}

}


void _napalm_export_exceptions()
{
	bp::class_<NapalmError> exc("NapalmError", bp::init<std::string>());
	exc.def( "__str__", &NapalmError::what);
	bp::register_exception_translator<NapalmError>(&translate);
	g_excType = exc.ptr();

	ExcDerivedBind<NapalmFileError>("NapalmFileError");
	ExcDerivedBind<NapalmSerializeError>("NapalmSerializeError");
}
