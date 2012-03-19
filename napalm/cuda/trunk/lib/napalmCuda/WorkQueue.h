

// our includes
#include "Singleton.h"

// boost includes
#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/condition.hpp>
#include <boost/thread/future.hpp>
#include <threadpool/threadpool.hpp>

namespace napalm
{

struct WorkQueue : public napalm::Singleton< WorkQueue >
{
	friend class napalm::Singleton< WorkQueue >;

	template< typename Func >
	boost::shared_future< typename boost::result_of< Func() >::type > submitJobAsync( Func f )
	{
		return submitJobImpl( m_pool, f );
	}

	template< typename Func >
	typename boost::result_of< Func() >::type submitJobSync( Func f )
	{
		boost::shared_future< typename boost::result_of< Func() >::type > future = submitJobImpl( m_pool, f );
		while( !future.is_ready() ){
			boost::thread::yield();
		}
		return future.get();
	}

private:
	WorkQueue( int workers = 1 )
	: m_pool( workers )
	{
		std::cerr << "constructing a workqueue\n";
	}

	template< typename Thp, typename Func >
	boost::shared_future< typename boost::result_of< Func() >::type > submitJobImpl( Thp& thp, Func f )
	{
		typedef typename boost::result_of< Func() >::type result;
		typedef boost::packaged_task< result > packaged_task;
		typedef boost::shared_ptr< boost::packaged_task< result > > packaged_task_ptr;

		packaged_task_ptr task( new packaged_task( f ) );
		boost::shared_future< result > res( task->get_future() );
		boost::threadpool::schedule( thp, boost::bind( &packaged_task::operator(), task ) );

		return res;
	}

	boost::threadpool::pool m_pool;
};

}


/***
    Copyright 2008-2012 Dr D Studios Pty Limited (ACN 127 184 954) (Dr. D Studios)

    This file is part of anim-studio-tools.

    anim-studio-tools is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    anim-studio-tools is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with anim-studio-tools.  If not, see <http://www.gnu.org/licenses/>.
***/
