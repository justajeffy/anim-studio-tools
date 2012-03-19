#include "napalmGL/context.h"

#include <boost/python.hpp>


using namespace boost::python;

BOOST_PYTHON_MODULE(_napalmGL)
{
	def( "hasOpenGL", napalm_gl::hasOpenGL );
	def( "noErrorsGL", napalm_gl::noErrorsGL );
}
