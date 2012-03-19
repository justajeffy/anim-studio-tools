#ifndef _NAPALM_PODBIND__H_
#define _NAPALM_PODBIND__H_

#include <boost/python.hpp>
#include "util/pod_wrapper.hpp"
#include "util/to_string.hpp"


namespace napalm
{

	namespace bp = boost::python;

	template<typename T>
	struct PodBind
	{
		typedef util::pod_wrapper<T> pod_wrapper_type;
		typedef typename pod_wrapper_type::arg_type arg_type;

		static std::string to_string( pod_wrapper_type & self )
		{
			return util::to_string<pod_wrapper_type>::value( self.get() );
		}

		PodBind(const std::string& name)
		{
			bp::class_<pod_wrapper_type, boost::shared_ptr<pod_wrapper_type> >(name.c_str(), bp::init<arg_type>())
				.add_property("value", &pod_wrapper_type::get, &pod_wrapper_type::set)
				.def("__str__", to_string)
				.def("__repr__", to_string);
				;
		}
	};

	template<>
	struct PodBind<half>
	{
		typedef util::pod_wrapper<half> pod_wrapper_type;

		static float get( pod_wrapper_type & a_Self )
		{
			return static_cast<float>( a_Self.get() );
		}

		static void set( pod_wrapper_type & a_Self, float a_Value )
		{
			a_Self.set(a_Value);
		}

		static std::string to_string( pod_wrapper_type & self )
		{
			return util::to_string<pod_wrapper_type>::value( self.get() );
		}

		PodBind(const std::string& name)
		{
			bp::class_<pod_wrapper_type, boost::shared_ptr<pod_wrapper_type> >(name.c_str(), bp::init<float>())
				.add_property("value", &get, &set)
				.def("__str__", to_string)
				.def("__repr__", to_string );
				;
		}
	};

}

#endif
