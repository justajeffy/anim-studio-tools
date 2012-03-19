#include <boost/python.hpp>
#include <boost/python/pure_virtual.hpp>
#include <boost/noncopyable.hpp>
#include "Attribute.h"
#include "Table.h"
#include "Buffer.h"


using namespace napalm;
namespace bp = boost::python;

// implementation note: placing BufferWrap into anon namespace causes internal compiler
// error running: gcc (GCC) 4.1.2 20080704 (Red Hat 4.1.2-44)
/*
struct BufferWrap : Buffer, bp::wrapper<Buffer>
{
	BufferWrap(store_ptr store):Buffer(store){}

	store_ptr getStore(bool readOnly) {
		return this->get_override("getStore")(readOnly);
	}

	bool setStore(store_ptr store, bool pullData) {
		return this->get_override("setStore")(store, pullData);
	}
};
*/

namespace {

	void resize1(Buffer& self, unsigned int n) {
		self.resize(n, false);
	}

	void resize2(Buffer& self, unsigned int n, bool destructive) {
		self.resize(n, destructive);
	}

}

void _napalm_export_Buffer()
{
	attrib_table_ptr (Buffer::*fn_getAttribs)() = &Buffer::getAttribs;

	//bp::class_<BufferWrap, bp::bases<Object>, boost::noncopyable >("Buffer", bp::no_init)
	bp::class_<Buffer, bp::bases<Object>, boost::noncopyable >("Buffer", bp::no_init)
		.def("__len__", &Buffer::size)
		.def("resize", resize1)
		.def("resize", resize2)
		.def("shrink", &Buffer::shrink)
		.def("clientSize", &Buffer::clientSize)
		.def("uniqueStore", &Buffer::uniqueStore)
		.def("storeUseCount", &Buffer::storeUseCount)
		.def("hasSharedStore", &Buffer::hasSharedStore)
		.add_property("attribs", fn_getAttribs, &Buffer::setAttribs)
		//.def("getStore", bp::pure_virtual(&Buffer::getStore))
		//.def("setStore", bp::pure_virtual(&Buffer::setStore))
		;
}






