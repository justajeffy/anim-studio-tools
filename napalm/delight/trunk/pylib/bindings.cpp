#include "napalmDelight/render.h"
#include "napalmDelight/io.h"

#include <napalm/core/Table.h>

#include <boost/python.hpp>


using namespace boost::python;

BOOST_PYTHON_MODULE(_napalmDelight)
{
	def( "sphere", napalm_delight::sphere );
	def( "points", napalm_delight::points );
	def( "curves", napalm_delight::curves );
	def( "load", napalm_delight::load );
	def( "save", napalm_delight::save );
}
