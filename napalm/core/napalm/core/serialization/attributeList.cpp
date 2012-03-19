#include "_serialization.h"
#include "../List.h"

namespace napalm {
void export_type_AttributeList()
{
	_EXPORT_TYPE( napalm::TypedAttribute<napalm::AttributeList>, AttributeListAttrib )
}
}
