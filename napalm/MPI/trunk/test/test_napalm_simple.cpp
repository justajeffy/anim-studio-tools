
// http://www.boost.org/doc/libs/1_37_0/doc/html/mpi/tutorial.html



#include <napalm/core/types/_types.h>
#include <napalm/core/core.h>
#include <napalm/core/io.h>

#include <boost/mpi.hpp>
#include <iostream>

namespace mpi = boost::mpi;
using namespace napalm;


int main( 	int argc, char* argv[] )
{
	mpi::environment env( argc, argv );
	mpi::communicator world;


	// test to see whether serialization is working at all...
	IntBufferPtr b = IntBufferPtr(new IntBuffer(10, 55));
	save( b, "/tmp/out.nap" );


	// send a string object to thyself
	if( 1 )
	{
		mpi::request reqs[ 2 ];

		std::string msg;
		std::string out_msg("hello world");

		reqs[ 0 ] = world.isend( 0, 0, out_msg );
		reqs[ 1 ] = world.irecv( 0, 0, msg );
		mpi::wait_all( reqs, reqs + 2 );

		std::cout << world.rank() << ": " << msg << std::endl;
	}


	// send a napalm object to thyself
	if( 1 )
	{
		mpi::request reqs[ 2 ];

		object_table_ptr msg(new ObjectTable());
		object_table_ptr out_msg(new ObjectTable());

		out_msg->setEntry("val", 5.0f );

		reqs[ 0 ] = world.isend( 0, 0, *out_msg );
		reqs[ 1 ] = world.irecv( 0, 0, *msg );
		mpi::wait_all( reqs, reqs + 2 );
		//std::cout << world.rank() << ": " << msg << std::endl;
	}

	return 0;
}

