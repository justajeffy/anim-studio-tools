#include "_typesBind.h"

void _napalm_export_vec2()
{
	_NAPALM_TYPE_BIND_OP(Imath::Vec2<int>)
	_NAPALM_TYPE_BIND_OP(Imath::Vec2<float>)
	_NAPALM_TYPE_BIND_OP(Imath::Vec2<double>)
	_NAPALM_TYPE_BIND_OP(Imath::Vec2<half>)
}
