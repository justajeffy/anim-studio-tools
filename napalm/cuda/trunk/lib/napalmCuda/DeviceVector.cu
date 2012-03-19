

#include "DeviceVector.h"

#include <thrust/fill.h>

namespace napalm
{

#define _NAPALM_CUDA_TYPE_OP( T ) \
template<> \
void DeviceVector<T>::fill( const T& val ) \
{ \
	thrust::fill( begin(), end(), val ); \
}
#include "base_cuda_types.inc"
#undef _NAPALM_CUDA_TYPE_OP

}
