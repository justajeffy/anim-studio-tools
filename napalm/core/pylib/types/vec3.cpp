#include "_typesBind.h"

void _napalm_export_vec3()
{
	_NAPALM_TYPE_BIND_OP(Imath::Vec3<int>)
	_NAPALM_TYPE_BIND_OP(Imath::Vec3<float>)
	_NAPALM_TYPE_BIND_OP(Imath::Vec3<double>)
	_NAPALM_TYPE_BIND_OP(Imath::Vec3<half>)
}
