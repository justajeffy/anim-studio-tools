#ifndef _NAPALM_TYPE_OP
#include "_types.h"
#include <bimath/matrix.hpp>
#define _NAPALM_TYPE_OP(T, Label) _INSTANTIATE_NAPALM_TYPE(T, Label)
#define __NAPALM_UNDEF
#endif
#include "type_enable.h"

#ifdef ENABLE_NAPALM_TYPES_IMATH
#ifdef ENABLE_NAPALM_TYPES_MATRIX
#ifdef ENABLE_NAPALM_TYPES_MATRIX33

#ifdef ENABLE_NAPALM_TYPES_BASED_ON_DOUBLE
_NAPALM_TYPE_OP(Imath::Matrix33<double>,		M33d)
#endif

#endif
#endif
#endif

#ifdef __NAPALM_UNDEF
#undef _NAPALM_TYPE_OP
#undef __NAPALM_UNDEF
#endif
