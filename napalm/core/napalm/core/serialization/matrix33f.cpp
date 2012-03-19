#include <bimath/matrix.hpp>
#include "_serialization.h"
#define _NAPALM_TYPE_OP(T, Label) _EXPORT_TYPES(T, Label)
#include "../types/type_enable.h"

#ifdef ENABLE_NAPALM_TYPES_IMATH
#ifdef ENABLE_NAPALM_TYPES_MATRIX
#ifdef ENABLE_NAPALM_TYPES_MATRIX33

#ifdef ENABLE_NAPALM_TYPES_BASED_ON_FLOAT
_NAPALM_TYPE_OP(Imath::Matrix33<float>,			M33f)
#endif

#endif
#endif
#endif

#undef _NAPALM_TYPE_OP
