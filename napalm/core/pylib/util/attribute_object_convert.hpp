#ifndef ATTRIBUTE_OBJECT_CONVERT_H
#define ATTRIBUTE_OBJECT_CONVERT_H

#include <boost/python.hpp>
#include "TypedAttribute.h"
#include "pod_wrapper.hpp"
#include <list>
#include "List.h"


namespace napalm { namespace util {
	namespace bp = boost::python;
	class ConversionDispatcher {
	public:
		typedef Attribute * (*convert_to_attribute_fn)(const bp::object &);
		typedef Attribute * (*convert_to_list_fn)(const bp::list &, bool );

		struct entry {
			convert_to_attribute_fn fn;
			convert_to_attribute_fn verify_fn;
			convert_to_list_fn construct_list;
		};
		typedef std::map<const PyTypeObject *, entry> map_type;

		static ConversionDispatcher& instance();

		ConversionDispatcher()
		{
#define _NAPALM_TYPE_OP(F, N) \
		AddPythonMapping<F>();\
		AddSetMapping<F>();
#include "../../napalm/core/types/all.inc"
#undef _NAPALM_TYPE_OP
		AddBoolMapping();
		AddVectorMapping();
		AddListMapping();
		}

		template<typename T>
		static typename boost::disable_if<
				boost::mpl::contains<napalm_wrapped_base_types,T>, Attribute * >::type
		MakeAttribute( const bp::object & a_Object )
		{
			bp::extract<T> extract(a_Object);
			if( extract.check() )
			{
				return new TypedAttribute<T>(extract());
			}
			else
			{
				return NULL;
			}
		}

		template<typename T>
		static typename boost::enable_if<
				boost::mpl::contains<napalm_wrapped_base_types,T>, Attribute * >::type
		MakeAttribute( const bp::object & a_Object )
		{
			bp::extract<pod_wrapper<T> > extract(a_Object);
			if( extract.check() )
			{
				return new TypedAttribute<T>(extract().get());
			}
			else
			{
				return NULL;
			}
		}

		template<typename T>
		static typename boost::disable_if<
				boost::mpl::contains<napalm_wrapped_base_types,T>, Attribute * >::type
				MakeAttributeVerified( const bp::object & a_Object )
		{
			bp::extract<T> extract(a_Object);
			if( extract.check() )
			{
				bp::object o = bp::object(T());
				if( o.ptr()->ob_type != a_Object.ptr()->ob_type )
				{
					// Necessary to prevent floats being read as ints etc.
					return NULL;
				}
				return new TypedAttribute<T>(extract());
			}
			else
			{
				return NULL;
			}
		}

		template<typename T>
		static typename boost::enable_if<
				boost::mpl::contains<napalm_wrapped_base_types,T>, Attribute * >::type
				MakeAttributeVerified( const bp::object & a_Object )
		{
			bp::extract<pod_wrapper<T> > extract(a_Object);
			if( extract.check() )
			{
				return new TypedAttribute<T>(extract().get());
			}
			else
			{
				return NULL;
			}
		}

		template<typename T>
		static Attribute * MakeSetAttribute( const bp::object & a_Object )
		{
			bp::extract<TypedAttribute<T> > extract(a_Object);
			if( extract.check() )
			{
				return new TypedAttribute<T>( extract().value() );
			}
			else
			{
				return NULL;
			}
		}

		static attrib_ptr AttribFromPython( const bp::object & a_Object )
		{
			map_type::iterator it = instance().m_Map.find( a_Object.ptr()->ob_type );
			if( it != instance().m_Map.end() )
			{
				return attrib_ptr( it->second.fn(a_Object) );
			}

			for( std::list<entry>::iterator vit = instance().m_Unallocated.begin();
					vit != instance().m_Unallocated.end(); ++vit )
			{
				Attribute * rv = vit->verify_fn(a_Object);
				if( rv )
				{
					instance().m_Map[a_Object.ptr()->ob_type] = *vit;
					instance().m_Unallocated.erase(vit);
					return attrib_ptr( rv );
				}
			}

			std::ostringstream strm;
			strm << "Could not find appropriate attribute conversion for object of type "<< a_Object.ptr()->ob_type->tp_name;
			PyErr_SetString(PyExc_KeyError, strm.str().c_str());
			bp::throw_error_already_set();

			return attrib_ptr();
		}

		template<typename T>
		static typename boost::disable_if<
				boost::mpl::contains<napalm_wrapped_base_types,T>, Attribute * >::type
				ConstructVector( const bp::list & list, bool pedantic )
		{
			boost::python::ssize_t n = boost::python::len(list);
			std::vector<T, util::counted_allocator<T> > rv;
			for(boost::python::ssize_t i=0;i<n;i++) {
				boost::python::object elem = list[i];
				bp::extract<T> extract(elem);
				if( extract.check() )
				{
					if( pedantic )
					{
						// This check is to prevent castable types (i.e. int->float)
						// being misinterpreted.
						bp::object o = bp::object(T());
						if( o.ptr()->ob_type != elem.ptr()->ob_type )
						{
							return NULL;
						}
					}
					rv.push_back(extract());
				}
				else
				{
					if(i != 0)
					{
						std::ostringstream strm;
						strm << "Mixed-type lists are not supported as attributes or keys!";
						PyErr_SetString(PyExc_KeyError, strm.str().c_str());
						bp::throw_error_already_set();
					}
					return NULL;
				}
		    }
			return new TypedAttribute<std::vector< T, util::counted_allocator<T> > >( rv );
		}

		template<typename T>
		static typename boost::enable_if<
				boost::mpl::contains<napalm_wrapped_base_types,T>, Attribute * >::type
				ConstructVector( const bp::list & list, bool pedantic )
		{
			boost::python::ssize_t n = boost::python::len(list);
			std::vector<T, util::counted_allocator<T> > rv;
			for(boost::python::ssize_t i=0;i<n;i++) {
				boost::python::object elem = list[i];
				bp::extract<pod_wrapper<T> > extract(elem);
				if( extract.check() )
				{
					rv.push_back(extract().get());
				}
				else
				{
					if(i != 0)
					{
						std::ostringstream strm;
						strm << "Mixed-type lists are not supported as attributes or keys!";
						PyErr_SetString(PyExc_KeyError, strm.str().c_str());
						bp::throw_error_already_set();
					}
					return NULL;
				}
		    }
			return new TypedAttribute<std::vector<T, util::counted_allocator<T> > >( rv );
		}

		static Attribute * AttribFromPyList( const bp::object & a_Object )
		{
			const bp::list * list = static_cast<const bp::list *>( &a_Object );
			if( !list )
			{
				return NULL;
			}
			// Treat empty lists as int lists.
			if( bp::len(*list) == 0 )
			{
				return new TypedAttribute<std::vector<int, util::counted_allocator<int> > >(std::vector<int, util::counted_allocator<int> >());
			}
			bp::object elem = (*list)[0];
			map_type::iterator it = instance().m_Map.find( elem.ptr()->ob_type );
			if( it != instance().m_Map.end() )
			{
				return it->second.construct_list(*list, false);
			}
			for( std::list<entry>::iterator vit = instance().m_Unallocated.begin();
					vit != instance().m_Unallocated.end(); ++vit )
			{
				Attribute * rv = vit->construct_list(*list, true);
				if( rv )
				{
					instance().m_Map[elem.ptr()->ob_type] = *vit;
					instance().m_Unallocated.erase(vit);
					return rv;
				}
			}

			std::ostringstream strm;
			strm << "Could not find appropriate attribute conversion for elements of type "<<elem.ptr()->ob_type->tp_name;
			PyErr_SetString(PyExc_KeyError, strm.str().c_str());
			bp::throw_error_already_set();

			return NULL;
		}

		static Attribute * AttribFromTuple( const bp::object & a_Object )
		{
			const bp::tuple * tuple = static_cast<const bp::tuple *>( &a_Object );
			if( !tuple )
			{
				return NULL;
			}
			TypedAttribute<AttributeList> * rv = new TypedAttribute<AttributeList>();

			int len = bp::len( *tuple );
			for( int i = 0; i < len; ++i )
			{
				rv->value().push_back( attrib_ptr( AttribFromPython( (*tuple)[i] ) ) );
			}

			return rv;
		}

		static Attribute * ConstructVectorError( const bp::list & list, bool pedantic )
		{
			std::ostringstream strm;
			strm << "Nested lists are not supported as attributes or keys!";
			PyErr_SetString(PyExc_KeyError, strm.str().c_str());
			bp::throw_error_already_set();
			return NULL;
		}

		static Attribute * ConstructBoolVectorError( const bp::list & list, bool pedantic )
		{
			std::ostringstream strm;
			strm << "Bool lists are not supported as attributes or keys!";
			PyErr_SetString(PyExc_KeyError, strm.str().c_str());
			bp::throw_error_already_set();
			return NULL;
		}

		template<typename T>
		void AddPythonMapping()
		{
			m_Unallocated.push_back( entry() );
			m_Unallocated.back().fn = &MakeAttribute<T>;
			m_Unallocated.back().verify_fn = &MakeAttributeVerified<T>;
			m_Unallocated.back().construct_list = &ConstructVector<T>;
		}

		template<typename T>
		void AddSetMapping()
		{
			m_Unallocated.push_back( entry() );
			m_Unallocated.back().fn = &MakeSetAttribute< std::set<T, util::less<T>, util::counted_allocator<T> > >;
			m_Unallocated.back().verify_fn = &MakeSetAttribute< std::set<T, util::less<T>, util::counted_allocator<T> > >;
			m_Unallocated.back().construct_list = &ConstructVectorError;
		}

		void AddBoolMapping()
		{
			m_Unallocated.push_back( entry() );
			m_Unallocated.back().fn = &MakeAttribute<bool>;
			m_Unallocated.back().verify_fn = &MakeAttributeVerified<bool>;
			m_Unallocated.back().construct_list = &ConstructBoolVectorError;
		}

		void AddVectorMapping()
		{
			bp::list l;
			m_Map[l.ptr()->ob_type].fn = &AttribFromPyList;
			m_Map[l.ptr()->ob_type].verify_fn = &AttribFromPyList;
			m_Map[l.ptr()->ob_type].construct_list = &ConstructVectorError;
		}

		void AddListMapping()
		{
			bp::tuple t;
			m_Map[t.ptr()->ob_type].fn = &AttribFromTuple;
			m_Map[t.ptr()->ob_type].verify_fn = &AttribFromTuple;
			m_Map[t.ptr()->ob_type].construct_list = &ConstructVectorError;
		}

	private:
		std::list<entry> m_Unallocated;
		map_type m_Map;
	};
} }

#endif
