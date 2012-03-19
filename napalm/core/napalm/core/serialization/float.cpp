#include "_serialization.h"
#define _NAPALM_TYPE_OP(T, Label) _EXPORT_TYPES(T, Label)
#include "../types/type_enable.h"

_NAPALM_TYPE_OP(float, 								Float)

#undef _NAPALM_TYPE_OP
