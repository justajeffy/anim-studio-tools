#include "_typesBind.h"

void _napalm_export_vec4()
{
	_NAPALM_TYPE_BIND_OP(Imath::Vec4<int>)
	_NAPALM_TYPE_BIND_OP(Imath::Vec4<float>)
	_NAPALM_TYPE_BIND_OP(Imath::Vec4<double>)
	_NAPALM_TYPE_BIND_OP(Imath::Vec4<half>)
}
