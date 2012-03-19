#ifndef _NAPALM_EXCEPTIONSBIND__H_
#define _NAPALM_EXCEPTIONSBIND__H_

#include <boost/python.hpp>
#include <boost/python/exception_translator.hpp>
#include "exceptions.h"


namespace napalm
{

	namespace bp = boost::python;


	template<typename T>
	struct ExcDerivedBind
	{
		ExcDerivedBind(const char* name)
		{
			bp::class_<T, bp::bases<NapalmError> > exc(name, bp::init<std::string>());
			bp::register_exception_translator<T>(&translate);
			m_excType = exc.ptr();
		}

		static void translate(const T& e)
		{
			bp::object pythonExceptionInstance(e);
			PyErr_SetObject(m_excType, pythonExceptionInstance.ptr());
		}

		static PyObject* m_excType;
	};

	template<typename T>
	PyObject* ExcDerivedBind<T>::m_excType(NULL);

}


#endif












