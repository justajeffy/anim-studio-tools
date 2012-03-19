



#include "DeviceVector.h"
#include "WorkQueue.h"
#include <cuda_runtime_api.h>

using namespace napalm;

/* A note on threading from /drd/software/packages/cuda/3.2.16/lin64/ext/doc/CUDA_C_Programming_Guide.pdf
 * Section 3.2
 *
 * Once the runtime has been initialized in a host thread, any resource (memory,
 * stream, event, etc.) allocated via some runtime function call in the host thread is
 * only valid within the context of the host thread. Therefore only runtime functions
 * calls made by the host thread (memory copies, kernel launches, …) can operate on
 * these resources. This is because a CUDA context (see Section 3.3.1) is created under
 * the hood as part of initialization and made current to the host thread, and it cannot
 * be made current to any other host thread.
 */

// currently disabled to allow usage of the workqueue at a higher level
// and to avoid having to transfer all details of DeviceVector to cpp
#define ENABLE_WORK_QUEUE 0

cudaError_t napalm::safeCudaMalloc( void ** devPtr, size_t sz )
{
#if ENABLE_WORK_QUEUE
	std::cerr << "allocating using WorkQueue" << std::endl;
	return WorkQueue::instance().submitJobSync( boost::bind( &cudaMalloc, devPtr, sz ) );
#else
	return cudaMalloc( devPtr, sz );
#endif
}

cudaError_t napalm::safeCudaFree( void* devPtr )
{
#if ENABLE_WORK_QUEUE
	return WorkQueue::instance().submitJobSync( boost::bind( &cudaFree, devPtr ) );
#else
	return cudaFree( devPtr );
#endif
}

