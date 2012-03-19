#include "TableBind.hpp"

void _napalm_export_AttribTable()
{
	using namespace napalm;
	TableBind<Attribute>("AttribTable");
}
