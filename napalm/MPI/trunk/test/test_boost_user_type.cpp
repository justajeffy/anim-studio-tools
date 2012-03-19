
// http://www.boost.org/doc/libs/1_37_0/doc/html/mpi/tutorial.html

#include <boost/mpi.hpp>
#include <iostream>
#include <boost/serialization/string.hpp>

namespace mpi = boost::mpi;

class gps_position
{
private:
	friend class boost::serialization::access;

	template< class Archive >
	void serialize( Archive & ar,
					const unsigned int version )
	{
		ar & degrees;
		ar & minutes;
		ar & seconds;
	}

	int degrees;
	int minutes;
	float seconds;
public:
	gps_position()
	{
	}
	;
	gps_position( 	int d,
					int m,
					float s ) :
		degrees( d ), minutes( m ), seconds( s )
	{
	}

	friend std::ostream& operator<< ( std::ostream& os, const gps_position& p )
	{
		return os << p.degrees << "," << p.minutes << "," << p.seconds;
	}

};




int main( 	int argc, char* argv[] )
{
	mpi::environment env( argc, argv );
	mpi::communicator world;

#if 0

	{
		mpi::request reqs[ 2 ];
		gps_position msg, out_msg( 1, 2, 3 );
		reqs[ 0 ] = world.isend( 0, 0, out_msg );
		reqs[ 1 ] = world.irecv( 0, 0, msg );
		mpi::wait_all( reqs, reqs + 2 );
		std::cout << world.rank() << ": " << msg << std::endl;
	}

# else

	if ( world.size() != 2 )
	{
		std::cout << "must have 2 ranks" << std::endl;
		return 0;
	}

	if ( world.rank() == 0 )
	{
		mpi::request reqs[ 2 ];
		gps_position msg, out_msg( 1, 2, 3 );
		reqs[ 0 ] = world.isend( 1, 0, out_msg );
		reqs[ 1 ] = world.irecv( 1, 1, msg );
		mpi::wait_all( reqs, reqs + 2 );
		std::cout << world.rank() << ": " << msg << std::endl;
	}
	else
	{
		mpi::request reqs[ 2 ];
		gps_position msg, out_msg( 2, 4, 6 );
		reqs[ 0 ] = world.isend( 0, 1, out_msg );
		reqs[ 1 ] = world.irecv( 0, 0, msg );
		mpi::wait_all( reqs, reqs + 2 );
		std::cout << world.rank() << ": " << msg << std::endl;
	}
#endif

	return 0;
}

