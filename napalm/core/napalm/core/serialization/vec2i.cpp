#include <bimath/vec.hpp>
#include "_serialization.h"
#define _NAPALM_TYPE_OP(T, Label) _EXPORT_TYPES(T, Label)
#include "../types/type_enable.h"

#ifdef ENABLE_NAPALM_TYPES_IMATH
#ifdef ENABLE_NAPALM_TYPES_VEC
#ifdef ENABLE_NAPALM_TYPES_VEC2

#ifdef ENABLE_NAPALM_TYPES_BASED_ON_INT
_NAPALM_TYPE_OP(Imath::Vec2<int>,				V2i)
#endif

#endif
#endif
#endif

#undef _NAPALM_TYPE_OP
