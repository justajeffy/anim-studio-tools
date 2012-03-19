#include "TableBind.hpp"


namespace bp = boost::python;


namespace napalm
{
void ToKeyPath( bp::object in, std::vector<attrib_ptr> & out )
{
	out.resize( bp::len( in ) );

	for( unsigned int i = 0; i < out.size(); ++i )
	{
		out[i] = util::ConversionDispatcher::AttribFromPython(in[i]);
	}
}

void raiseKeyError(const TableKey& key)
{
	std::ostringstream strm;
	strm << *key << " is not in table";
	PyErr_SetString(PyExc_KeyError, strm.str().c_str());
	bp::throw_error_already_set();
}

void raiseKeyError(const std::vector<attrib_ptr> & key)
{
	std::ostringstream strm;
	strm << "Path: ( ";
	for ( std::vector<attrib_ptr>::const_iterator it = key.begin();
			it != key.end(); ++it )
	{
		if( it != key.begin() )
		{
			strm << ", ";
		}
		strm << **it;
	}
	strm << " ) is not in table";
	PyErr_SetString(PyExc_KeyError, strm.str().c_str());
	bp::throw_error_already_set();
}

void raiseKeyError(const bp::object & object)
{
	std::ostringstream strm;
	strm << object << " is not a valid key";
	PyErr_SetString(PyExc_KeyError, strm.str().c_str());
	bp::throw_error_already_set();
}

}










