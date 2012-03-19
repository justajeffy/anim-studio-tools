#include <bimath/box.hpp>
#include <bimath/vec.hpp>
#include "_serialization.h"
#define _NAPALM_TYPE_OP(T, Label) _EXPORT_TYPES(T, Label)
#include "../types/type_enable.h"

#ifdef ENABLE_NAPALM_TYPES_IMATH
#ifdef ENABLE_NAPALM_TYPES_BOX
#ifdef ENABLE_NAPALM_TYPES_BOX2

#ifdef ENABLE_NAPALM_TYPES_BASED_ON_FLOAT
_NAPALM_TYPE_OP(Imath::Box<Imath::Vec2<float> >,	B2f)
#endif

#endif
#endif
#endif

#undef _NAPALM_TYPE_OP
