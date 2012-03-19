#ifndef _NAPALM_TYPE_OP
#include "_types.h"
#include <bimath/box.hpp>
#define _NAPALM_TYPE_OP(T, Label) _INSTANTIATE_NAPALM_TYPE(T, Label)
#define __NAPALM_UNDEF
#endif
#include "type_enable.h"

#ifdef ENABLE_NAPALM_TYPES_IMATH
#ifdef ENABLE_NAPALM_TYPES_BOX
#ifdef ENABLE_NAPALM_TYPES_BOX3

#ifdef ENABLE_NAPALM_TYPES_BASED_ON_INT
_NAPALM_TYPE_OP(Imath::Box<Imath::Vec3<int> >,		B3i)
#endif

#endif
#endif
#endif

#ifdef __NAPALM_UNDEF
#undef _NAPALM_TYPE_OP
#undef __NAPALM_UNDEF
#endif
