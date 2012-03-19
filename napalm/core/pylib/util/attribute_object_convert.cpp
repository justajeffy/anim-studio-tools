#include "attribute_object_convert.hpp"

using namespace napalm::util;

ConversionDispatcher& ConversionDispatcher::instance()
{
	static ConversionDispatcher inst;
	return inst;
}
