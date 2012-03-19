#include <OpenEXR/ImathVec.h>
#include <OpenEXR/ImathBox.h>
#include <boost/random.hpp>
#include <spatials/kdtree.hpp>
#include <adaptors/points.hpp>
#include <points.hpp>
#include <vector>
#include <time.h>
#include <fstream>
#include <sys/time.h>

//-----------------------------------------------------------------------------
class SimpleTimer
{
	//-----------------------------------------------------------------------------
public:
	SimpleTimer()
	{
		gettimeofday( &m_TimeVal, &m_TimeZone );
	}
	float Elapsed() const
	{
		struct timeval tv;
		struct timezone tz;
		gettimeofday( &tv, &tz );
		float now = ( tv.tv_sec - m_TimeVal.tv_sec ) + 0.000001f * ( tv.tv_usec - m_TimeVal.tv_usec );
		return now;
	}

	//-----------------------------------------------------------------------------
private:
	struct timeval m_TimeVal;
	struct timezone m_TimeZone;
};


//-----------------------------------------------------------------------------
template< class point_storage >
void
Dump( const typename dgal::KDTree< point_storage >::Node & n, int depth )
{
	if ( n.isLeaf() )
	{
		for ( typename dgal::KDTree< point_storage >::IndexIterator it = n.getFirst() ;
			it != n.getLast() ; ++it )
		{
			//for (int i = depth; i-->0;) //std::cout << "  ";
			//std::cout << *it << " : " << n->getAdaptor()[ *it ] << std::endl;
		}
		//std::cout << std::endl;
	}
	else
	{
		//for (int i = depth; i-->0;) //std::cout << "  ";
		//std::cout << *n->getMid() << " : " << n->getAdaptor()[ *n->getMid() ] << " / " << n->getAxis() << " (" << 
		//	n->getAdaptor()[ *n->getMid() ][n->getAxis()] << ")" << std::endl;

		//for (int i = depth; i-->0;) //std::cout << "  ";
		//std::cout << "B:" << std::endl;
		Dump< point_storage >( n.getBefore(), depth + 2 );


		//for (int i = depth; i-->0;) //std::cout << "  ";
		//std::cout << "A:" << std::endl;
		Dump< point_storage >( n.getAfter(), depth + 2 );
	}
}

//-----------------------------------------------------------------------------
template< class point_type >
void
TestKD( int count, typename point_type::BaseType range, int Add, int s )
{
	typedef std::vector< point_type > point_storage;
	typedef typename point_storage::iterator point_iter;
	std::vector< boost::mt19937 > randGens;
	std::vector< boost::variate_generator<boost::mt19937, boost::uniform_real<typename point_type::BaseType> > > nDs;

	boost::uniform_real<typename point_type::BaseType> uni_dist(-range, range);

	for ( unsigned int d = 0 ; d < point_type::dimensions() ; ++d )
	{
		randGens.push_back( boost::mt19937( static_cast<unsigned int>( /*time(0) +*/ d + count +s ) ) );
		nDs.push_back( boost::variate_generator<boost::mt19937, boost::uniform_real<typename point_type::BaseType> > (randGens.back(), uni_dist) );
	}

	point_storage points;
	for ( int i = 0 ; i < count ; ++i )
	{
		point_type p;
		for ( unsigned int d = 0 ; d < point_type::dimensions() ; ++d )
			p[ d ] = nDs[d]();
		points.push_back( p );
	}

	// the actual KDtree
	SimpleTimer kdtreecreate;
	dgal::KDTree< point_storage > dkdtree( points, Add );
	float kdtreecreateelapsed = kdtreecreate.Elapsed();

	Imath::Box< point_type > qBox;
	for ( int i = 0 ; i < 2 ; ++i )
	{
		point_type p;
		for ( unsigned int d = 0 ; d < point_type::dimensions() ; ++d )
			p[ d ] = nDs[d]();
		qBox.extendBy( p );
	}

	{
		typename dgal::KDTree< point_storage >::Index outResult;
		SimpleTimer kdtreequery;
		dkdtree.query( qBox, outResult );
		float kdtreequeryelapsed = kdtreequery.Elapsed();

		SimpleTimer boxquery;
		typename dgal::KDTree< point_storage >::Index outResultBrute;
		for ( point_iter it = points.begin() ; it != points.end() ; ++it )
		{
			if ( qBox.intersects(*it) ) 
				outResultBrute.push_back( it-points.begin() );
		}

		float boxqueryelapsed = boxquery.Elapsed();
		std::cout << "  Create: " << kdtreecreateelapsed*1000.0f << "ms kd: " << kdtreequeryelapsed*1000.0f << "ms brute: " 
			<< " " << boxqueryelapsed*1000.0f << " "
			<< (kdtreequeryelapsed/boxqueryelapsed)*100.0f << "% : "
			<< "ms, : " << outResult.size() << " items" << std::endl;
	}
	{
		point_type qp;
		for ( unsigned int d = 0 ; d < point_type::dimensions() ; ++d )
			qp[ d ] = nDs[d]();
		typename dgal::KDTree< point_storage >::Index outResultKd;

		typename point_type::BaseType dist = 2;
		typename point_type::BaseType distSqr = dist*dist;

		SimpleTimer kdtreequery;
		dkdtree.query( qp, dist, outResultKd );
		float kdtreequeryelapsed = kdtreequery.Elapsed();

		typename dgal::KDTree< point_storage >::Index outResultBrute;

		SimpleTimer brutetreequery;
		for ( point_iter it = points.begin() ; it != points.end() ; ++it )
		{
			if ( (qp-(*it)).length2() < distSqr ) 
				outResultBrute.push_back( it-points.begin() );
		}
		float brutetreequeryelapsed = brutetreequery.Elapsed();

		std::cout << "    kd: " << kdtreequeryelapsed*1000.0f << "ms, " << brutetreequeryelapsed*1000.0f << "ms : "	
			<< (kdtreequeryelapsed/brutetreequeryelapsed)*100.0f << "% : "
			<< outResultKd.size() << "/" << outResultBrute.size() <<  " items" << std::endl;
	}
}

//-----------------------------------------------------------------------------
int
main( int argc, char ** argv )
{
	int count = 1600000;
	for ( int j = 0 ; j < 1 ; ++j )
	{
		for ( int i = 0 ; i < 8 ; ++i )
		{
			std::cout << "Testing on: " << (4+i*4) << " : " << j << std::endl;
			TestKD<Imath::V2f>( count, 100.0f, 4+i*4, j );
			TestKD<Imath::V3d>( count, 100.0,  4+i*4, j );
		}
		std::cout << std::endl;
	}
	return 0;
}
