#ifndef _NAPALM_TYPE_INFO__H_
#define _NAPALM_TYPE_INFO__H_

#include <typeinfo>
#include <cxxabi.h>
#include <string>
#include <cassert>


namespace napalm { namespace util {

	// return an std::string containing the demangled name of the type
	inline std::string get_type_name(const char* mangled_name)
	{
		char* tstr = abi::__cxa_demangle(mangled_name, 0, 0, 0);
		std::string tstring(tstr);
		free(tstr);
		return tstring;
	}

	template<typename T>
	std::string get_type_name() {
		return get_type_name(typeid(T).name());
	}

	// compare functor so type_info can be used as key in std::map
	struct type_info_compare
	{
		bool operator()(const std::type_info* a, const std::type_info* b) {
			assert(a && b);
			return a->before(*b);
		}
	};

} } // ns

#endif
