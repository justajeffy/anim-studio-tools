/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: $"
 * SVN_META_ID = "$Id: $"
 */

//-------------------------------------------------------------------------------------------------
#include "timer.h"
#include "cuda.h"

namespace grind
{

	GPUTimer::GPUTimer()
	{
	  e_start = new cudaEvent_t;
	  e_stop = new cudaEvent_t;

	  cudaEventCreate((cudaEvent_t *)e_start);
	  cudaEventCreate((cudaEvent_t *)e_stop);
	}

	GPUTimer::~GPUTimer()
	{
	  cudaEventDestroy(*((cudaEvent_t *)e_start));
	  cudaEventDestroy(*((cudaEvent_t *)e_stop));

	  delete (cudaEvent_t *)e_start;
	  delete (cudaEvent_t *)e_stop;
	}

	void GPUTimer::start() {
	  cudaEventRecord(*((cudaEvent_t *)e_start), 0);
	}

	void GPUTimer::stop()  {
	  cudaEventRecord(*((cudaEvent_t *)e_stop), 0);
	}

	float GPUTimer::elapsed_ms()
	{
	    cudaEventSynchronize(*((cudaEvent_t *)e_stop));
	    float ms;
	    cudaEventElapsedTime(&ms, *((cudaEvent_t *)e_start), *((cudaEvent_t *)e_stop));
	    return ms;
	}

} // grind
