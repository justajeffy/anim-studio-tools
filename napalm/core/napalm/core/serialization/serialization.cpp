#include "_serialization.h"

namespace napalm {
#define _NAPALM_INCLUDE_LIST
#define _NAPALM_INCLUDE_BOOL
#define _NAPALM_TYPE_OP( T, Label ) void export_type_##Label();
#include "../types/all.inc"
#undef _NAPALM_TYPE_OP
#undef _NAPALM_INCLUDE_BOOL
#undef _NAPALM_INCLUDE_LIST
}

void SetupSerialization()
{
#define _NAPALM_INCLUDE_LIST
#define _NAPALM_INCLUDE_BOOL
#define _NAPALM_TYPE_OP( T, Label ) napalm::export_type_##Label();
#include "../types/all.inc"
#undef _NAPALM_TYPE_OP
#undef _NAPALM_INCLUDE_BOOL
#undef _NAPALM_INCLUDE_LIST
}
