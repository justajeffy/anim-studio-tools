#include <boost/python.hpp>

extern void _napalm_export_free_functions();
extern void _napalm_export_exceptions();
extern void _napalm_export_pod_wrappers();
extern void _napalm_export_Object();
extern void _napalm_export_Buffer();
extern void _napalm_export_ObjectTable();
extern void _napalm_export_AttribTable();

extern void _napalm_export_basic();
extern void _napalm_export_box2();
extern void _napalm_export_box3();
extern void _napalm_export_matrix33();
extern void _napalm_export_matrix44();
extern void _napalm_export_vec2();
extern void _napalm_export_vec3();
extern void _napalm_export_vec4();


BOOST_PYTHON_MODULE(_napalm_core)
{
	_napalm_export_free_functions();
	_napalm_export_exceptions();
	_napalm_export_pod_wrappers();
	_napalm_export_Object();
	_napalm_export_Buffer();
	_napalm_export_ObjectTable();
	_napalm_export_AttribTable();

	_napalm_export_basic();
	_napalm_export_box2();
	_napalm_export_box3();
	_napalm_export_matrix33();
	_napalm_export_matrix44();
	_napalm_export_vec2();
	_napalm_export_vec3();
	_napalm_export_vec4();
}






