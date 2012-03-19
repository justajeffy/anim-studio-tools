#include "_typesBind.h"

void _napalm_export_box3()
{
	_NAPALM_TYPE_BIND_OP(Imath::Box<Imath::Vec3<int> >)
	_NAPALM_TYPE_BIND_OP(Imath::Box<Imath::Vec3<float> >)
	_NAPALM_TYPE_BIND_OP(Imath::Box<Imath::Vec3<double> >)
	_NAPALM_TYPE_BIND_OP(Imath::Box<Imath::Vec3<half> >)
}
