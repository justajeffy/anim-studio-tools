#ifndef _H5CPP_UTIL_TYPE_INFO__H_
#define _H5CPP_UTIL_TYPE_INFO__H_

#include <typeinfo>
#include <cxxabi.h>
#include <string>


namespace h5cpp { namespace util {


	// return an std::string containing the demangled name of the type
	template<typename T>
	std::string get_type_name()
	{
		char* tstr = abi::__cxa_demangle(typeid(T).name(), 0, 0, 0);
		std::string tstring(tstr);
		free(tstr);
		return tstring;
	}


	// compare functor so type_info can be used as key in std::map
	struct type_info_compare
	{
		bool operator()(const std::type_info* a, const std::type_info* b)
		{
			assert(a && b);
			return a->before(*b);
		}
	};

} } // ns

#endif
