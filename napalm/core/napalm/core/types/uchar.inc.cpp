#ifndef _NAPALM_TYPE_OP
#include "_types.h"
#define _NAPALM_TYPE_OP(T, Label) _INSTANTIATE_NAPALM_TYPE(T, Label)
#define __NAPALM_UNDEF
#endif

_NAPALM_TYPE_OP(unsigned char, 						UChar)

#ifdef __NAPALM_UNDEF
#undef _NAPALM_TYPE_OP
#undef __NAPALM_UNDEF
#endif
