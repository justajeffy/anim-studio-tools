
// http://www.boost.org/doc/libs/1_37_0/doc/html/mpi/tutorial.html

#include <boost/mpi.hpp>
#include <iostream>
#include <boost/serialization/string.hpp>
namespace mpi = boost::mpi;

int main( 	int argc, char* argv[] )
{
	mpi::environment env( argc, argv );
	mpi::communicator world;

	if ( world.size() != 2 )
	{
		std::cout << "must have 2 ranks" << std::endl;
		return 0;
	}

	if ( world.rank() == 0 )
	{
		mpi::request reqs[ 2 ];
		std::string msg, out_msg = "Hello";
		reqs[ 0 ] = world.isend( 1, 0, out_msg );
		reqs[ 1 ] = world.irecv( 1, 1, msg );
		mpi::wait_all( reqs, reqs + 2 );
		std::cout << msg << "!" << std::endl;
	}
	else
	{
		mpi::request reqs[ 2 ];
		std::string msg, out_msg = "world";
		reqs[ 0 ] = world.isend( 0, 1, out_msg );
		reqs[ 1 ] = world.irecv( 0, 0, msg );
		mpi::wait_all( reqs, reqs + 2 );
		std::cout << msg << ", ";
	}

	return 0;
}

