#ifndef _NAPALM_UTIL_POD_WRAPPER__H_
#define _NAPALM_UTIL_POD_WRAPPER__H_

#include <string>
#include <boost/mpl/vector.hpp>
#include <boost/numeric/conversion/cast.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/mpl/contains.hpp>
#include "util/to_string.hpp"
#include "type_convert.hpp"


namespace napalm { namespace util {


	typedef boost::mpl::vector<
		char,
		unsigned char,
		short,
		unsigned short,
		unsigned int,
		double,
		half
	> napalm_wrapped_base_types;


	template<typename T>
	struct pod_wrapper
	{
		typedef type_converter<T> conv_type;
		typedef typename conv_type::type arg_type;

		pod_wrapper(arg_type t){ set(t); }
		arg_type get() const { return conv_type::to_python(m_value); }
		void set(const arg_type t) { m_value = conv_type::from_python(t); }

		T m_value;
	};


	template<typename T>
	std::ostream& operator <<(std::ostream& os, const pod_wrapper<T>& pod)
	{
		os << util::to_string<T>::value(pod.m_value);
		return os;
	}

	// type traits
	template<typename T, typename Enable=void>
	struct pod_traits
	{
		typedef T value_type;
		static inline const T& get_value(const value_type& val) { return val; }
	};

	template<typename T>
	struct pod_traits<T, typename boost::enable_if<
		boost::mpl::contains<napalm_wrapped_base_types,T> >::type>
	{
		typedef pod_wrapper<T> value_type;
		static inline const T& get_value(const value_type& val) { return val.m_value; }
	};

} } // ns


#endif







