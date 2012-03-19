#include "system.h"
#include "io.h"
#include "types/_types.h"
#include "core.h"
#include "typedefs.h"
#include "List.h"


extern void SetupSerialization();

namespace napalm {

template class TypedAttribute<bool>;
void _initialise_napalm_base_type_Bool()
{
	_INIT_NAPALM_TYPE(TypedAttribute<bool>, BoolAttrib);
	Dispatcher::instance().AddSelfOnlyMapRow<bool>();
}
template class TypedAttribute<AttributeList>;
void _initialise_napalm_base_type_AttributeList()
{
	_INIT_NAPALM_TYPE(TypedAttribute< AttributeList >, ListAttrib);
	Dispatcher::instance().AddSelfOnlyMapRow< AttributeList >();
}

template< typename T >
void misc_napalm_type( const char * name )
{
	detail::initialise_napalm_type<T> (name);
	::boost::serialization::singleton<
		::boost::archive::detail::guid_initializer<T>
			>::get_mutable_instance().export_guid(name);
}

#define _NAPALM_INCLUDE_BOOL
#define _NAPALM_INCLUDE_LIST
#define _NAPALM_TYPE_OP(T, Label) \
	extern void _initialise_napalm_base_type_##Label();
	#include "types/all.inc"
#undef _NAPALM_TYPE_OP
#undef _NAPALM_INCLUDE_LIST
#undef _NAPALM_INCLUDE_BOOL


_napalm_init::_napalm_init()
{
	NapalmSystem::instance().init();
	NapalmSystem::count_bytes(0);
}


NapalmSystem& NapalmSystem::instance()
{
	static NapalmSystem inst;
	return inst;
}


void NapalmSystem::init()
{
	static bool initialised = false;
	if(initialised)
		return;

	m_version = BOOST_PP_STRINGIZE(VERSION);
	sscanf(m_version.c_str(), "%d.%d.%d", &m_majorVersion, &m_minorVersion, &m_patchVersion);

	m_totalClientBytes = 0;

	#define _NAPALM_INCLUDE_BOOL
	#define _NAPALM_INCLUDE_LIST
	#define _NAPALM_TYPE_OP(T, Label) _initialise_napalm_base_type_##Label();
	#include "types/all.inc"
	#undef _NAPALM_TYPE_OP
	#undef _NAPALM_INCLUDE_LIST
	#undef _NAPALM_INCLUDE_BOOL

	SetupSerialization();

	misc_napalm_type<Object>					("Object");
	misc_napalm_type<Attribute>					("Attribute");
	misc_napalm_type<Buffer>					("Buffer");
	misc_napalm_type<BufferStore>				("BufferStore");
	misc_napalm_type<detail::NapalmFileHeader>	("NapalmFileHeader");
	misc_napalm_type<BufferStoreHolder>			("BufferStoreHolder");
	misc_napalm_type<ObjectTable>				("ObjectTable");
	misc_napalm_type<AttribTable>				("AttribTable");

	initialised = true;
}


void NapalmSystem::count_bytes(long b)
{
	static long* totalClientBytes = &(instance().m_totalClientBytes);
	util::add_and_fetch(*totalClientBytes, b);
}

}

