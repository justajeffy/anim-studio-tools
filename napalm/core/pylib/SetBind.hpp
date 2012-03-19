#include <set>
#include <boost/python.hpp>
#include <boost/python/pure_virtual.hpp>
#include <boost/python/stl_iterator.hpp>
#include <boost/noncopyable.hpp>
#include "Object.h"
#include "util/attribute_object_convert.hpp"
#include "TypedAttribBind.hpp"
#include "util/less_than.hpp"


using namespace napalm;


template<typename T>
static bp::list toList( const TypedAttribute< std::set<T, util::less<T>, util::counted_allocator<T> > >& a_Self )
{
	bp::list rv;
	typename std::set<T, util::less<T>, util::counted_allocator<T> >::const_iterator it = a_Self.value().begin();
	for( ; it != a_Self.value().end(); ++it )
	{
		rv.append( util::type_converter<T>::to_python( *it ) );
	}
	return rv;
}

template<typename T>
static bp::tuple toTuple( const TypedAttribute< std::set<T, util::less<T>, util::counted_allocator<T> > >& a_Self )
{
	return bp::tuple( toList( a_Self ) );
}

template<typename T>
static int size( const TypedAttribute< std::set<T, util::less<T>, util::counted_allocator<T>  > >& a_Self )
{
	return a_Self.value().size();
}

template<typename T>
static bp::object pop( TypedAttribute< std::set<T, util::less<T>, util::counted_allocator<T>  > >& a_Self, int a_Index )
{
	if( a_Self.value().size() == 0 )
	{
		return bp::object();
	}

	typename std::set<T, util::less<T> >::const_iterator it = a_Self.value().begin();
	bp::object rv ( util::type_converter<T>::to_python( *it ) );
	a_Self.value().erase(it);
	return rv;
}

template<typename T>
static void add( TypedAttribute< std::set<T, util::less<T>, util::counted_allocator<T>  > > & a_Self, bp::object a_Ele )
{
	bp::extract<T> get_obj( a_Ele );
	if( get_obj.check() )
	{
		a_Self.value().insert( get_obj() );
	}
}

template<typename T>
static TypedAttribute< std::set<T, util::less<T>, util::counted_allocator<T> > >* objectInit(const bp::object& x)
{
	TypedAttribute< std::set<T, util::less<T>, util::counted_allocator<T> > >* rv = new TypedAttribute< std::set<T, util::less<T>, util::counted_allocator<T>  > >();

	for(bp::stl_input_iterator<bp::object> it( x ), end; it != end; ++it )
	{
		add( *rv, *it );
	}
	return rv;
}

template<typename T>
static bool contains( TypedAttribute< std::set<T, util::less<T>, util::counted_allocator<T> > > & a_Self, bp::object a_Ele )
{
	bp::extract<T> get_obj( a_Ele );
	if( get_obj.check() )
	{
		return a_Self.value().find( get_obj() ) != a_Self.value().end();
	}
	return false;
}

template<typename T>
static typename std::set<T, util::less<T>, util::counted_allocator<T> >::iterator set_begin( TypedAttribute< std::set<T, util::less<T>, util::counted_allocator<T> > > & a_Self )
{
	return a_Self.value().begin();
}

template<typename T>
static typename std::set<T, util::less<T>, util::counted_allocator<T> >::iterator set_end( TypedAttribute< std::set<T, util::less<T>, util::counted_allocator<T> > > & a_Self )
{
	return a_Self.value().end();
}

template<typename T>
void SetBind(const std::string & name)
{
	bp::class_< TypedAttribute< std::set<T, util::less<T>, util::counted_allocator<T> > >, bp::bases<Object> >(name.c_str())
		.def("__init__", bp::make_constructor(objectInit<T>) )
		.def("__len__", size<T> )
		.def("pop", pop<T> )
		.def("toList", toList<T> )
		.def("toTuple", toTuple<T> )
		.def( "add", add<T> )
		.def( "__contains__", contains<T> )
		.def("__iter__", bp::range( set_begin<T>, set_end<T> ) );
		;
}






