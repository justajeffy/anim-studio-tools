
/*
 * Base types used by napalm. To add new types, append to the relevant cpp file in this
 * directory, or create a new cpp file following the same structure as the others, and
 * add it to the include list here.
 */

#ifndef _NAPALM_TYPE_OP
#error "expected _NAPALM_TYPE_OP to be defined"
#endif

#include "char.inc.cpp"
#include "uchar.inc.cpp"
#include "short.inc.cpp"
#include "ushort.inc.cpp"
#include "int.inc.cpp"
#include "uint.inc.cpp"
#include "float.inc.cpp"
#include "double.inc.cpp"
#include "half.inc.cpp"
#include "string.inc.cpp"

#include "vec2i.inc.cpp"
#include "vec2f.inc.cpp"
#include "vec2d.inc.cpp"
#include "vec2h.inc.cpp"

#include "vec3i.inc.cpp"
#include "vec3f.inc.cpp"
#include "vec3d.inc.cpp"
#include "vec3h.inc.cpp"

#include "vec4i.inc.cpp"
#include "vec4f.inc.cpp"
#include "vec4d.inc.cpp"
#include "vec4h.inc.cpp"

#include "matrix33f.inc.cpp"
#include "matrix33d.inc.cpp"

#include "matrix44f.inc.cpp"
#include "matrix44d.inc.cpp"

#include "box2i.inc.cpp"
#include "box2f.inc.cpp"
#include "box2d.inc.cpp"
#include "box2h.inc.cpp"

#include "box3i.inc.cpp"
#include "box3f.inc.cpp"
#include "box3d.inc.cpp"
#include "box3h.inc.cpp"

// These last two are only used in certain cases.
#include "bool.inc"
#include "attribList.inc"






