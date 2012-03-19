#ifndef _NAPALM_TABLEBIND__H_
#define _NAPALM_TABLEBIND__H_

#include <boost/python.hpp>
#include <boost/python/iterator.hpp>
#include <boost/mpl/for_each.hpp>
#include <boost/utility/enable_if.hpp>
#include <vector>
#include <stdexcept>
#include <sstream>
#include "Table.h"
#include "TypedAttribute.h"
#include "TypedAttribBind.hpp"
#include "meta.hpp"
#include "util/pod_wrapper.hpp"
#include "util/attribute_object_convert.hpp"


namespace napalm
{
	namespace bp = boost::python;

	void raiseKeyError(const TableKey& key);
	void raiseKeyError(const std::vector<attrib_ptr>& key);
	void raiseKeyError(const bp::object & object);

	void ToKeyPath( bp::object in, std::vector<attrib_ptr> & out );


	template<typename Value>
	struct TableBind
	{
		typedef Table<Value> 												table_type;
		typedef typename table_type::value_ptr								table_value_ptr;
		typedef boost::shared_ptr<table_type> 								table_type_ptr;
		typedef bp::class_<table_type, bp::bases<Object>, table_type_ptr> 	bp_class_;


		// return converter for table keys
		struct copy_key
		{
			template <class T>
			struct apply
			{
				typedef typename table_type::value_type value_type;
				typedef value_type result_converter;

				struct type
				{
					const PyTypeObject* get_pytype() { return 0; }
					bool convertible() const { return true; }
					typedef PyObject* result_type;

					PyObject* operator()(const value_type& p) const
					{
						bp::object key = attrib_topython::convert(p.first);
						return bp::incref(key.ptr());
					}
				};
			};
		};

		// return converter for table values
		struct copy_value
		{
			template <class T>
			struct apply
			{
				typedef typename table_type::value_type value_type;
				typedef value_type result_converter;

				struct type
				{
					const PyTypeObject* get_pytype() { return 0; }
					bool convertible() const { return true; }
					typedef PyObject* result_type;

					PyObject* operator()(const value_type& p) const
					{
						bp::object key = attrib_topython::convert(p.first);
						bp::object value = attrib_topython::convert(p.second);
						bp::tuple x = bp::make_tuple(key, value);
						return bp::incref(x.ptr());
					}
				};
			};
		};


		// '__contains__' method binding
		static bool _contains(const table_type& self, const bp::object& key)
		{
			attrib_ptr attrib ( util::ConversionDispatcher::AttribFromPython(key ) );
			if( attrib == NULL )
			{
				return false;
			}
			TableKey tk( attrib );
			return self.hasEntry( tk );
		}


		// '__del__' method binding
		static void _del(table_type& self, const bp::object& key)
		{
			attrib_ptr attrib ( util::ConversionDispatcher::AttribFromPython(key ) );
			if( attrib == NULL )
			{
				return;
			}
			TableKey tk( attrib );
			if(!self.delEntry(tk))
				raiseKeyError(tk);
		}


		// '__setitem__' method binding
		void bind_setitem(bp_class_ cl)
		{
		    // This order is significant as setattribitem will accept the input
		    // that setobjectitem wants, but can't process it meaningfully.
			cl.def("__setitem__", _setvalueitem);
			cl.def("__setitem__", _setobjectitem);
		}

		static void _setobjectitem(table_type& self, const bp::object& key, table_value_ptr value)
		{
			attrib_ptr attrib ( util::ConversionDispatcher::AttribFromPython(key ) );
			if( attrib == NULL )
			{
				std::ostringstream strm;
				strm << "Could not find appropriate attribute conversion for key";
				PyErr_SetString(PyExc_KeyError, strm.str().c_str());
				bp::throw_error_already_set();
				return;
			}
			TableKey tk( attrib );
			self.setEntry( tk, value );
		}

		static void _setvalueitem(table_type& self, const bp::object & k,
		                                  const bp::object & v)
		{
			attrib_ptr aKey ( util::ConversionDispatcher::AttribFromPython(k) );
			attrib_ptr aValue ( util::ConversionDispatcher::AttribFromPython(v) );
			if( aKey && aValue )
			{
				self.setEntry( TableKey( aKey ), aValue );
			}
			else if( !aKey )
			{
				std::ostringstream strm;
				strm << "Could not find appropriate attribute conversion for key";
				PyErr_SetString(PyExc_KeyError, strm.str().c_str());
				bp::throw_error_already_set();
			}
			else
			{
				std::ostringstream strm;
				strm << "Could not find appropriate attribute conversion for value";
				PyErr_SetString(PyExc_KeyError, strm.str().c_str());
				bp::throw_error_already_set();
			}
		}

		static void _setitembypath_value(table_type& self, bp::object obj, table_value_ptr value, bool shouldCreate )
		{
			std::vector<attrib_ptr> key;
			ToKeyPath( obj, key );
			if( !self.setEntryPath( key, value, shouldCreate ) )
			{
				std::ostringstream strm;
				strm << "Parent of key path does not exist, or cannot contain this object";
				PyErr_SetString(PyExc_KeyError, strm.str().c_str());
				bp::throw_error_already_set();
			}
		}

		static void _setitembypath(table_type& self, bp::object obj,
										  const bp::object & v, bool shouldCreate )
		{
			std::vector<attrib_ptr> k;
			ToKeyPath( obj, k );
			attrib_ptr aValue ( util::ConversionDispatcher::AttribFromPython(v) );
			if( aValue )
			{
				if( !self.setEntryPath( k, aValue, shouldCreate ) )
				{
					std::ostringstream strm;
					strm << "Parent of key path does not exist, or cannot contain this object";
					PyErr_SetString(PyExc_KeyError, strm.str().c_str());
					bp::throw_error_already_set();
				}
			}
			else
			{
				std::ostringstream strm;
				strm << "Could not find appropriate attribute conversion for value";
				PyErr_SetString(PyExc_KeyError, strm.str().c_str());
				bp::throw_error_already_set();
			}
		}

		static void _setitembypath_value_noc(table_type& self, bp::object obj, table_value_ptr value)
		{
			_setitembypath_value(self, obj, value, false);
		}
		static void _setitembypath_noc(table_type& self, bp::object obj, const bp::object & v)
		{
			_setitembypath(self, obj, v, false);
		}


		// '__getitem__' method binding
		static bp::object _getitem(const table_type& self, const bp::object& key)
		{
			attrib_ptr attrib( util::ConversionDispatcher::AttribFromPython(key) );
			if( attrib == NULL )
			{
				std::ostringstream strm;
				strm << "Could not find appropriate attribute conversion for key";
				PyErr_SetString(PyExc_KeyError, strm.str().c_str());
				bp::throw_error_already_set();
				return bp::object();
			}

			object_ptr obj;
			TableKey tk( attrib );
			if(!self.getEntry( tk, obj))
				raiseKeyError( tk );

			return attrib_topython::convert(obj);
		}

		// '__getitem__' method binding - for keypaths
		static bp::object _getitembypath(const table_type& self, const bp::object keypath)
		{
			object_ptr obj;
			std::vector<attrib_ptr> key;
			ToKeyPath( keypath, key );
			if(!self.getEntryPath( key, obj))
				raiseKeyError( key );

			return attrib_topython::convert(obj);
		}
		// '__delitem__' method binding - for keypaths
		static void _delitembypath(table_type& self, const bp::object keypath)
		{
			std::vector<attrib_ptr> key;
			ToKeyPath( keypath, key );
			if(!self.delEntryPath( key))
				raiseKeyError( key );
		}
		// 'realvalue' method binding
		static bp::object _realvalue(const table_type& self, const bp::object& key)
		{
			attrib_ptr attrib( util::ConversionDispatcher::AttribFromPython(key) );
			if( attrib == NULL )
			{
				std::ostringstream strm;
				strm << "Could not find appropriate attribute conversion for key";
				PyErr_SetString(PyExc_KeyError, strm.str().c_str());
				bp::throw_error_already_set();
				return bp::object();
			}

			object_ptr obj;
			TableKey tk( attrib );
			if(!self.getEntry( tk, obj))
				raiseKeyError( tk );

			return attrib_topython::convert(obj, true);
		}

		// '__hasitem__' method binding
		static bool _hasitem(const table_type& self, const bp::object& key)
		{
			attrib_ptr attrib( util::ConversionDispatcher::AttribFromPython(key) );
			if( attrib == NULL )
			{
				std::ostringstream strm;
				strm << "Could not find appropriate attribute conversion for key";
				PyErr_SetString(PyExc_KeyError, strm.str().c_str());
				bp::throw_error_already_set();
				return bp::object();
			}

			object_ptr obj;
			TableKey tk( attrib );
			return self.getEntry( tk, obj);
		}


		// 'keys' binding
		static bp::list keys(const table_type& self)
		{
			bp::list l;
			for(typename table_type::const_iterator it = self.begin(); it!=self.end(); ++it)
				l.append(attrib_topython::convert(it->first));
			return l;
		}

		// 'values' binding
		static bp::list values(const table_type& self)
		{
			bp::list l;
			for(typename table_type::const_iterator it = self.begin(); it!=self.end(); ++it)
				l.append(attrib_topython::convert(it->second));
			return l;
		}


		// 'items' binding
		static bp::list items(const table_type& self)
		{
			bp::list l;
			for(typename table_type::const_iterator it = self.begin(); it!=self.end(); ++it)
			{
				bp::object key = attrib_topython::convert(it->first);
				bp::object value = attrib_topython::convert(it->second);
				bp::tuple x = bp::make_tuple(key, value);
				l.append(x);
			}
			return l;
		}


		TableBind(const char* name)
		{
			bp_class_ cl(name);

			cl.def("empty", &table_type::empty);
			cl.def("clear", &table_type::clear);
			cl.def("keys", keys);
			cl.def("values", values);
			cl.def("items", items);
			cl.def("has_key", _hasitem);
			cl.def("__len__", &table_type::size);
			cl.def("__iter__", bp::iterator<table_type, bp::return_value_policy<copy_key> >());
			cl.def("iteritems", bp::iterator<table_type, bp::return_value_policy<copy_value> >());
			cl.def("__delitem__", _del);
			cl.def("__contains__", _contains);
			cl.def("__getitem__", _getitem);
			cl.def("set_by_path", _setitembypath);
			cl.def("set_by_path", _setitembypath_value);
			cl.def("set_by_path", _setitembypath_noc);
			cl.def("set_by_path", _setitembypath_value_noc);
			cl.def("get_by_path", _getitembypath);
			cl.def("del_by_path", _delitembypath);
			cl.def("realvalue", _realvalue);

			bind_setitem(cl);
		}

	};


}


#endif











