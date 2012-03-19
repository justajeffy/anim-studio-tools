#include <iostream>
#include <cutil_math.h>

//#include <napalm/util/fixed_range.hpp>

#include <napalmCuda/DeviceVector.h>

//! A custom functor
struct MyFunctor
{
	__host__ __device__
	void operator()( float& v )
	{
		v += 1.0f;
	}
};

void myGpuAlgorithm( napalm::DeviceVector<float>& src )
{
	std::cout << "running gpu algorithm on floats..." << std::endl;
	MyFunctor f;
	thrust::for_each( src.begin(), src.end(), f );
}

struct MyFunctor2
{
	__host__ __device__
	void operator()( float3& v )
	{
		v.x += 1.0f;
		v.y += 2.0f;
		v.z += 3.0f;
	}
};

void myGpuAlgorithm2( napalm::DeviceVector<float3>& src )
{
	std::cout << "running gpu algorithm on float3s" << std::endl;
	MyFunctor2 f;
	thrust::for_each( src.begin(), src.end(), f );
}

