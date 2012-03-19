
/*
 * Base types used by napalm. Note that order is important as it can affect python
 * bindings - int must appear _after_ float, otherwise integer arguments in python may
 * be passed to overloaded float-bound functions.
 *
 * To add new types, append to the end of this list, then add the relevant macros in
 * system.cpp.
 */

_NAPALM_CUDA_TYPE_OP(char)
_NAPALM_CUDA_TYPE_OP(short)

_NAPALM_CUDA_TYPE_OP(int)
_NAPALM_CUDA_TYPE_OP(int2)
_NAPALM_CUDA_TYPE_OP(int3)
_NAPALM_CUDA_TYPE_OP(int4)

_NAPALM_CUDA_TYPE_OP(float)
_NAPALM_CUDA_TYPE_OP(float2)
_NAPALM_CUDA_TYPE_OP(float3)
_NAPALM_CUDA_TYPE_OP(float4)

_NAPALM_CUDA_TYPE_OP(double)
