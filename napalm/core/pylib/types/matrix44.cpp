#include "_typesBind.h"

void _napalm_export_matrix44()
{
	_NAPALM_TYPE_BIND_OP(Imath::M44f)
	_NAPALM_TYPE_BIND_OP(Imath::M44d)
}
