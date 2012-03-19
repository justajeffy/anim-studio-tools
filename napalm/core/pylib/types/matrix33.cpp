#include "_typesBind.h"

void _napalm_export_matrix33()
{
	_NAPALM_TYPE_BIND_OP(Imath::M33f)
	_NAPALM_TYPE_BIND_OP(Imath::M33d)
}
