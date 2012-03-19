#ifndef _NAPALM_TYPEDATTRIBBIND__H_
#define _NAPALM_TYPEDATTRIBBIND__H_

#include <boost/python.hpp>
#include <boost/mpl/vector.hpp>
#include <map>
#include "TypedAttribute.h"
#include "impl/ValueWrapper.hpp"
#include "util/type_info.hpp"
#include "util/type_convert.hpp"
#include "util/pod_wrapper.hpp"
#include "List.h"


namespace napalm
{

	namespace bp = boost::python;

	/*
	 * To-python conversion static class
	 */

	template<typename T>
	typename boost::enable_if<boost::mpl::contains<util::napalm_wrapped_base_types,T>, bp::object>::type
	pod_if_needed( const T & value )
	{
		return bp::object( util::pod_wrapper<T>( value ) );
	}

	template<typename T>
	typename boost::disable_if<boost::mpl::contains<util::napalm_wrapped_base_types,T>, bp::object>::type
	pod_if_needed( const T & value )
	{
		return bp::object( value );
	}

	class attrib_topython
	{
	public:

		static bp::object convert(const object_ptr& o, bool a_Wrap = false );
		static bp::object convert(const attrib_ptr& a, bool a_Wrap = false );
		static bp::object convert(const Wrapper<Attribute>& a, bool a_Wrap = false )
		{
			return convert( a.get(), a_Wrap );
		}

		template<typename T>
		static bp::object convert_t(attrib_ptr a, bool a_Wrap)
		{
			typedef TypedAttribute<T> attrib_type;
			assert(dynamic_cast<attrib_type*>(a.get()));
			attrib_type* pa = reinterpret_cast<attrib_type*>(a.get());
			if( a_Wrap )
			{
				return pod_if_needed( util::type_converter<T>::to_python(pa->value()));
			}
			else
			{
				return bp::object(util::type_converter<T>::to_python(pa->value()));
			}
		}

		typedef bp::object (*converter_fn)(attrib_ptr, bool);
		typedef std::map<const std::type_info*, converter_fn, util::type_info_compare> map_type;

		static map_type s_map;
	};


	/*
	 * To-python conversion registration
	 */
	template<typename T>
	struct TypedAttribConvert
	{
		typedef TypedAttribute<T> attrib_type;

		TypedAttribConvert()
		{
			attrib_topython::converter_fn fn = attrib_topython::convert_t<T>;
			const std::type_info* pti = &typeid(attrib_type);
			attrib_topython::s_map.insert(attrib_topython::map_type::value_type(pti, fn));
		}
	};

	/*
	 * To-python conversion registration
	 */
	template<typename T>
	struct TypedAttribConvertVector
	{
		typedef TypedAttribute<std::vector<T, util::counted_allocator<T> > > attrib_type;

		static
		bp::object ConvertArray(attrib_ptr a, bool a_Wrap)
		{
			typedef TypedAttribute<std::vector<T, util::counted_allocator<T> > > attrib_type;
			assert(dynamic_cast<attrib_type*>(a.get()));
			attrib_type& at = *reinterpret_cast<attrib_type*>(a.get());
			boost::python::list l = boost::python::list();
			if( a_Wrap )
			{
				std::vector<T, util::counted_allocator<T> > & vec = at.value();
				for(size_t i = 0; i < vec.size(); i++)
					l.append(pod_if_needed<T>(util::type_converter<T>::to_python(vec[i])));
			}
			else
			{
				std::vector<T, util::counted_allocator<T> > & vec = at.value();
		        for(size_t i = 0; i < vec.size(); i++)
		            l.append(bp::object(util::type_converter<T>::to_python(vec[i])));
			}

	        return l;
		}

		TypedAttribConvertVector()
		{
			attrib_topython::converter_fn fn = ConvertArray;
			const std::type_info* pti = &typeid(attrib_type);
			attrib_topython::s_map.insert(attrib_topython::map_type::value_type(pti, fn));
		}
	};

	template<typename T>
	struct TypedAttribConvertSet
	{
		typedef TypedAttribute<std::set<T, util::less<T>, util::counted_allocator<T> > > attrib_type;

		static bp::object ConvertSet(attrib_ptr a, bool a_Wrap)
		{
			typedef TypedAttribute<std::set<T, util::less<T>, util::counted_allocator<T>  > > attrib_type;
			assert(dynamic_cast<attrib_type*>(a.get()));
			attrib_type& at = *reinterpret_cast<attrib_type*>(a.get());

			return bp::object(at);

		}

		TypedAttribConvertSet()
		{
			attrib_topython::converter_fn fn = ConvertSet;
			const std::type_info* pti = &typeid(attrib_type);
			attrib_topython::s_map.insert(attrib_topython::map_type::value_type(pti, fn));
		}
	};

	struct ListAttribConverter
	{
		typedef TypedAttribute< AttributeList > attrib_type;

		static bp::object ConvertList(attrib_ptr a, bool a_Wrap)
		{
			assert(dynamic_cast<attrib_type*>(a.get()));
			attrib_type& at = *reinterpret_cast<attrib_type*>(a.get());
			AttributeList & vec = at.value();
			bp::list vector = boost::python::list();
			for(size_t i = 0; i < vec.size(); i++)
			{
				vector.append(attrib_topython::convert(vec[i], a_Wrap));
			}

			return bp::tuple( vector );

		}

		ListAttribConverter()
		{
			attrib_topython::converter_fn fn = ConvertList;
			const std::type_info* pti = &typeid(attrib_type);
			attrib_topython::s_map.insert(attrib_topython::map_type::value_type(pti, fn));
		}
	};
}

#endif



