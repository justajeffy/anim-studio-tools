#include <boost/python.hpp>
#include <boost/noncopyable.hpp>
#include "io.h"
#include "system.h"
#include "TypedBuffer.h"
//#include "parsing/parsePythonDict.h"


using namespace napalm;
namespace bp = boost::python;

namespace {

	bool _areEqual(object_ptr obj1, object_ptr obj2) {
		return areEqual(obj1, obj2);
	}

	void _save1(object_ptr obj, const std::string& filepath) {
		SaveOptions op;
		save(obj, filepath, op);
	}

	void _save2(object_ptr obj, const std::string& filepath,
		unsigned int compression)
	{
		SaveOptions op(compression);
		save(obj, filepath, op);
	}

	CharBufferPtr _saveToMemory1(object_ptr obj) {
		return saveToMemory(obj);
	}

	CharBufferPtr _saveToMemory2(object_ptr obj, unsigned int compression)
	{
		SaveOptions op(compression);
		return saveToMemory(obj, op);
	}

	object_ptr _load1(const std::string& filepath) {
		return load(filepath);
	}

	static object_ptr _load2(const std::string& filepath, bool delayLoad)
	{
		LoadOptions op(delayLoad);
		return load(filepath, op);
	}

	object_ptr _loadFromMemory(CharBufferPtr buf) {
		return loadFromMemory(buf);
	}

	void _dump(object_ptr obj) {
		if( obj )
		{
			obj->dump(std::cout);
		}
	}

	const std::string& _getVersionString() {
		return NapalmSystem::instance().getVersionString();
	}

	long _getTotalClientBytes() {
		return NapalmSystem::instance().getTotalClientBytes();
	}

} // anon ns


void _napalm_export_free_functions()
{
	bp::def("areEqual", _areEqual);
	bp::def("dump", _dump);
	bp::def("getAPIVersion", _getVersionString, bp::return_value_policy<bp::copy_const_reference>());
	bp::def("getTotalClientBytes", _getTotalClientBytes);

	bp::def("save", _save1);
	bp::def("save", _save2);
	bp::def("saveToMemory", _saveToMemory1);
	bp::def("saveToMemory", _saveToMemory2);

	bp::def("load", _load1);
	bp::def("load", _load2);
	bp::def("loadFromMemory", _loadFromMemory);
}


















