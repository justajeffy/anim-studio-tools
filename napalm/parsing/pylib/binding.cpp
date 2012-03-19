#include <boost/python.hpp>
#include "parsePythonDict.h"
#include "parseXml.h"


using namespace napalm;
namespace bp = boost::python;


namespace {

	object_table_ptr _parseDict(const std::string& dict_str) {
		return parsePythonDict(dict_str);
	}

	object_table_ptr _parseXml(const std::string& dict_str) {
		return parseXml(dict_str);
	}

} // anon ns


BOOST_PYTHON_MODULE(_napalm_parsing)
{
	bp::def("parseDict", _parseDict);
	bp::def("parseXml", _parseXml);
}






