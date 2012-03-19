#ifndef _NAPALM_UTIL_GIL_WRAPPER__H_
#define _NAPALM_UTIL_GIL_WRAPPER__H_

#include <Python.h>


namespace napalm { namespace util {

	/*
	 * This file defines the function "gil_wrapper". When used to wrap a boost.python-bound
	 * function, the python GIL will be unlocked during function execution. Example:
	 *
	 * boost::python::def("big_cpp_func", util::gil_wrapper(big_cpp_func));
	 */


	/*
	 * @class scoped_gil_release
	 * @brief
	 * Class which unlocks the python GIL on construction, and locks it again on
	 * destruction. Used to free the GIL while executing C++ code bound to python.
	 */
	struct scoped_gil_release()
	{
		scoped_gil_release():m_tstate(PyEval_SaveThread()){}
		~scoped_gil_release(){PyEval_RestoreThread(m_tstate);}
		PyThreadState* m_tstate;
	};





} } // ns


#endif
