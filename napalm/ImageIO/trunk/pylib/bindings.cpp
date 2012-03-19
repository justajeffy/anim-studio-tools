#include "napalmImageIO/image.h"

#include <napalm/core/Table.h>

#include <boost/python.hpp>

using namespace boost::python;

// get around const binding problem

bool _isValid( 	napalm::object_table_ptr t,
				bool report = true )
{
	napalm_image_io::isValid( t, report );
}

void _write( 	napalm::object_table_ptr t,
				const std::string& filePath,
				const std::string& destFormat )
{
	napalm_image_io::write( t, filePath, destFormat );
}

BOOST_PYTHON_MODULE(_napalmImageIO)
{
	def( "isValid", _isValid );
	def( "write", _write );
	def( "read", napalm_image_io::read );
}
