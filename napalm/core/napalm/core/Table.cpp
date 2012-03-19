#include "Table.h"

namespace napalm {
	namespace detail {
		bool setAttribEntry(AttribTable& atable, const TableKey& key, const attrib_ptr value)
		{
			atable.setEntry(key, value);
			return true;
		}
	}
}
