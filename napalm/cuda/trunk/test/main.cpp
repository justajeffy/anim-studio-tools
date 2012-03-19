#define NAPALM_LOG_CPU_GPU_TRANSFERS 1
#define NAPALM_LOG_GPU_ALLOC 1

#include <boost/mpl/aux_/na.hpp>

#include <napalm/core/TypedAttribute.h>
#include <napalm/core/TypedBuffer.h>
#include <napalm/core/Table.h>

#include <napalmCuda/BufferStoreCuda.h>
#include <napalmCuda/cuda_type_logging.h>
#include <napalmCuda/WorkQueue.h>

#include <iostream>

#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>


#include "cutil_math.h"

using namespace napalm;

//! A gpu algorithm that we'll be calling on a device vector
void myGpuAlgorithm( napalm::DeviceVector< float >& src );
void myGpuAlgorithm2( napalm::DeviceVector< float3 >& src );

//! Dump something that supports standard array access
template< typename T >
void dump( const T& b )
{
	for( size_t i = 0; i < b.size(); ++i )
		std::cout << b[i] << " ";
	std::cout << std::endl;
}

void bufferTransferTest()
{
#if 1
	// try creating a cuda buffer store directly
	BufferStoreCuda< float > f( 100, 1.0f );
#endif


#if 1
	object_table_ptr t( new ObjectTable() );
	{
		// initialize the data
		FloatBufferPtr buf( new FloatBuffer( 20 ) );
		FloatBuffer::w_type fr = buf->w();
		fr[4] = 3.1415927;
		std::cout << "original values..." << std::endl;
		dump( fr );

		t->setEntry( "radius", buf );
	}

	{
		FloatBufferPtr b;
		t->getEntry("radius", b);

		// some gpu algo
		if(b)
		{
			// to be centralized
			store_ptr store = b->getStore( false );
			boost::shared_ptr<BufferStoreCuda< float > > thstore = boost::dynamic_pointer_cast< BufferStoreCuda< float > >(store);
			if(!thstore)
			{
				thstore.reset( new BufferStoreCuda< float >() );
				b->setStore(thstore, true);
			}
			assert( thstore );

			// access the actual buffer
			BufferStoreCuda<float>::cuda_buffer_type& c = thstore->buffer();

			// run a gpu algorithm
			myGpuAlgorithm( c );

			std::cout << "device vec contents..." << std::endl;
			dump( c );
		}
	}
	{
		FloatBufferPtr b;
		t->getEntry("radius", b);

		// another gpu algo
		if( b )
		{
			// to be centralized
			store_ptr store = b->getStore( false );
			boost::shared_ptr<BufferStoreCuda< float > > thstore = boost::dynamic_pointer_cast< BufferStoreCuda< float > >(store);
			if(!thstore)
			{
				thstore.reset( new BufferStoreCuda< float >() );
				b->setStore(thstore, true);
			}
			assert( thstore );

			// access the actual buffer
			BufferStoreCuda<float>::cuda_buffer_type& c = thstore->buffer();

			// run a gpu algorithm
			myGpuAlgorithm( c );

			std::cout << "device vec contents..." << std::endl;
			dump( c );
		}
	}
	{
		FloatBufferPtr b;
		t->getEntry("radius", b);

		// some cpu based algo
		if( b )
		{
			// conventional buffer access
			store_ptr store = b->getStore( false );
			FloatBufferPtr buf;
			t->getEntry( "radius", buf );
			FloatBuffer::r_type fr = buf->r();
			std::cout << "fixed_range values..." << std::endl;
			dump( fr );
		}
	}

#endif

#if 1
	{
		// initialize vector data
		V3fBufferPtr buf( new V3fBuffer( 10 ) );
		V3fBuffer::w_type fr = buf->w();
		fr[4] = Imath::V3f(3.1415927,3.1415927,3.1415927);
		std::cout << "original values..." << std::endl;
		dump( fr );

		t->setEntry( "P", buf );
	}

	{
		V3fBufferPtr b;
		t->getEntry("P", b);

		// some gpu algo
		if(b)
		{
			// to be centralized
			store_ptr store = b->getStore( false );
			boost::shared_ptr<BufferStoreCuda< Imath::V3f > > thstore = boost::dynamic_pointer_cast< BufferStoreCuda< Imath::V3f > >(store);
			if(!thstore)
			{
				thstore.reset( new BufferStoreCuda< Imath::V3f >() );
				b->setStore(thstore, true);
			}
			assert( thstore );

			// access the actual buffer
			BufferStoreCuda< Imath::V3f >::cuda_buffer_type& c = thstore->buffer();

			// run a gpu algorithm
			myGpuAlgorithm2( c );

			std::cout << "device vec contents..." << std::endl;
			dump( c );
		}

		// some cpu based algo
		if( b )
		{
			// conventional buffer access
			store_ptr store = b->getStore( false );
			V3fBufferPtr buf;
			t->getEntry( "P", buf );
			V3fBuffer::r_type fr = buf->r();
			std::cout << "fixed_range values..." << std::endl;
			dump( fr );
		}
	}


#endif

}

boost::mutex mtx;

struct ThreadTestBuf
{
	ThreadTestBuf( 	boost::shared_ptr< BufferStore >& b,
					int op ) :
		m_buf( b ), m_op( op )
	{
	}

	void processOp()
	{
		std::cout << "processOp() from thread id: " << ( unsigned int ) pthread_self() << std::endl;
		boost::mutex::scoped_lock s( mtx );
		boost::shared_ptr< BufferStoreCuda<float> > b;
		switch( m_op ){
			case 0:
				m_buf.reset( new BufferStoreCuda< float >( 20, 1.3f ) );
				b = boost::dynamic_pointer_cast<  BufferStoreCuda<float> >( m_buf );
				dump( b->buffer() );
				break;
			case 1:
				std::cout << "eval on thread id: " << ( unsigned int ) pthread_self() << std::endl;
				b = boost::dynamic_pointer_cast<  BufferStoreCuda<float> >( m_buf );
				assert( b );
				myGpuAlgorithm( b->buffer() );
				dump( b->buffer() );
				break;
			case 2:
				m_buf.reset();
				break;
			default:
				assert(0);
		}
	}

	void operator()()
	{
		std::cout << "operator() from thread id: " << ( unsigned int ) pthread_self() << std::endl;
		// do the actual processing via the WorkQueue
		WorkQueue::instance().submitJobSync( boost::bind( &ThreadTestBuf::processOp, this ) );
	}

	boost::shared_ptr< BufferStore >& m_buf;
	int m_op;
};

void threadTest()
{
	std::cout << "running thread test from thread id: " << ( unsigned int ) pthread_self() << std::endl;

	boost::shared_ptr< BufferStore > buf;

#if 0
	ThreadTestBuf c0( buf, 0 );
	c0();
	ThreadTestBuf c1( buf, 1 );
	c1();
	ThreadTestBuf c2( buf, 2 );
	c2();
#else

	ThreadTestBuf c0( buf, 0 );
	boost::thread t0(c0);

	usleep(50);

	ThreadTestBuf c1( buf, 1 );
	boost::thread t1(c1);

	usleep(50);

	ThreadTestBuf c2( buf, 2 );
	boost::thread t2(c2);

	t0.join();
	t1.join();
	t2.join();
#endif

}

int main( 	int arc,
			char** argv )
{
	std::cout << "\n--------------------------------------------------------" << std::endl;

	// do all work via the WorkQueue, including allocation, processing and deallocation
	WorkQueue::instance().submitJobSync( boost::bind( &bufferTransferTest ) );

	std::cout << "\n--------------------------------------------------------" << std::endl;
	threadTest();

	std::cout << "done" << std::endl;

	return 0;
}
