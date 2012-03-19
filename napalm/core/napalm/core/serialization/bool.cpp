#include "_serialization.h"

namespace napalm {
void export_type_Bool()
{
	_EXPORT_TYPE( napalm::TypedAttribute<bool>, BoolAttrib )
}
}
