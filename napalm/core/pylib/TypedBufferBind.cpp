#include "util/bimath.h"
#include "TypedBufferBind.hpp"

namespace napalm {

std::size_t verifiedIndex(std::size_t size, int index)
{
	std::size_t index_ = static_cast<std::size_t>(index);
	if((index < 0) || (index_ >= size))
	{
		std::ostringstream strm;
		strm << "index " << index << " out of range (buffer size=" << size << ')';
		throw std::out_of_range(strm.str());
	}
	return index_;
}

}
