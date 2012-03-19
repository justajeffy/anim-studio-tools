#include "TableBind.hpp"

void _napalm_export_ObjectTable()
{
	using namespace napalm;
	TableBind<Object>("ObjectTable");
}
