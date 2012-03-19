#include <boost/mpl/for_each.hpp>
#include "PodBind.hpp"
#include "typelabels.h"


using namespace napalm;

namespace {

	struct pod_wrapper_bind_visitor
	{
		template<typename T>
		void operator()(T x)
		{
			std::string label(type_label<T>::value());
			PodBind<T>(label+""); // todo move naming into podbind
		}
	};

}

void _napalm_export_pod_wrappers()
{
	boost::mpl::for_each<util::napalm_wrapped_base_types>(pod_wrapper_bind_visitor());
}
