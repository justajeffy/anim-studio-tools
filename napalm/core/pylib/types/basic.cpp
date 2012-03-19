#include "_typesBind.h"

void _napalm_export_basic()
{
	using namespace napalm;
	std::string label(type_label<bool>::value());
	TypedAttribConvert<bool>();
	ListAttribConverter();

	_NAPALM_TYPE_BIND_OP(float)
	_NAPALM_TYPE_BIND_OP(double)
	_NAPALM_TYPE_BIND_OP(char)
	_NAPALM_TYPE_BIND_OP(unsigned char)
	_NAPALM_TYPE_BIND_OP(short)
	_NAPALM_TYPE_BIND_OP(unsigned short)
	_NAPALM_TYPE_BIND_OP(int)
	_NAPALM_TYPE_BIND_OP(unsigned int)
	_NAPALM_TYPE_BIND_OP(half)
	_NAPALM_TYPE_BIND_OP(std::string)
}
