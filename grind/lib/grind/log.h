/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: $"
 * SVN_META_ID = "$Id: log.h 60899 2011-01-10 23:30:22Z chris.cooper $"
 */

#ifndef grind_log_h
#define grind_log_h

//-------------------------------------------------------------------------------------------------
#include <drdDebug/log.h>
#include <drdDebug/runtimeError.h>
#include <iostream>
#include <stdexcept>
#include <pthread.h>
#include <cassert>
#include "context.h"

namespace grind {

//! @cond DEV

//-------------------------------------------------------------------------------------------------
//! check error state and log with file/line if there's an error
#define checkErrorGL() \
{ \
	GLenum errCode; \
 \
	if ((errCode = glGetError()) != GL_NO_ERROR) { \
		DRD_LOG_ERROR( L, "** OpenGL Error at " << __FILE__ << ":" << __LINE__ << ": " << gluErrorString(errCode)); \
	   throw std::runtime_error( "OpenGL error" ); \
	} \
}

//-------------------------------------------------------------------------------------------------
#define requireContextGL() \
{ \
	if ( !grind::ContextInfo::instance().hasOpenGL() ){ \
		DRD_LOG_ERROR( L, "** OpenGL context required at " << __FILE__ << ":" << __LINE__ ); \
		throw std::runtime_error("OpenGL context required" ); \
	} \
}

//-------------------------------------------------------------------------------------------------
#define requireContextGPU() \
{ \
	if ( !grind::ContextInfo::instance().hasGPU() ){ \
		DRD_LOG_ERROR( L, "** GPU context required at " << __FILE__ << ":" << __LINE__ ); \
		throw std::runtime_error("GPU context required" ); \
	} \
}

//-------------------------------------------------------------------------------------------------
/*#ifdef DEBUG_BUILD */
#if 1
#define SAFE_GL( fn ) \
{ \
	requireContextGL(); \
	{fn;} \
	checkErrorGL(); \
}
#else
#define SAFE_GL( fn ) {fn;}
#endif

//-------------------------------------------------------------------------------------------------
/*#ifdef DEBUG_BUILD */
#if 1
#define SAFE_CUDA( fn ) \
{ \
	requireContextGPU(); \
	cudaError errCode; \
\
    if( (errCode = fn) != cudaSuccess ) { \
	   DRD_LOG_ERROR( L, "** CUDA Error at thread: " << ( unsigned int ) pthread_self() << " - " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString( errCode ) ); \
	   throw drd::RuntimeError( "CUDA error" ); \
    } \
}
#else
#define SAFE_CUDA( fn ) \
{ \
	cudaError errCode; \
\
    if( (errCode = fn) != cudaSuccess ) { \
    	DRD_LOG_ERROR( L, "** CUDA Error at thread: " << ( unsigned int ) pthread_self() << " - " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString( errCode ) ); \
	   throw drd::RuntimeError( "CUDA error" ); \
    } \
}
#endif


//-------------------------------------------------------------------------------------------------
//! log a variable name and value
#define LOGVAR(x) DRD_LOG_INFO( L, "var " << #x << ": " << x );

//-------------------------------------------------------------------------------------------------
//! log line number and file
#define LOGLINE() DRD_LOG_INFO( L, "* line " << __LINE__ << " in " << __FILE__ );


#ifndef NDEBUG
/*
 * function entry point logging
 * you must define __class__ eg...
 *
 * #define __class__ "MyClass"
 *
 * __func__ should be provided by the compiler
 *
 */
#define LOGFN0() std::cout << "C++:" << __class__ << ":" << __func__ << "()" << std::endl;
#define LOGFN1( v0 ) std::cout << "C++:" << __class__ << ":" << __func__<< "( " << #v0 << "=" << v0 << " )" << std::endl;
#define LOGFN2( v0, v1 ) std::cout << "C++:" << __class__ << ":" << __func__<< "( " << #v0 << "=" << v0 << "," << #v1 << "=" << v1 << " )" << std::endl;
#define LOGFN3( v0, v1, v2 ) std::cout << "C++:" << __class__ << ":" << __func__<< "( " << #v0 << "=" << v0 << "," << #v1 << "=" << v1  << "," << #v2 << "=" << v2 << " )" << std::endl;
#else
#define LOGFN0()
#define LOGFN1( v0 )
#define LOGFN2( v0, v1 )
#define LOGFN3( v0, v1, v2 )
#endif

//! @endcond

}

#endif /* grind_log_h_ */


/***
    Copyright 2008-2012 Dr D Studios Pty Limited (ACN 127 184 954) (Dr. D Studios)

    This file is part of anim-studio-tools.

    anim-studio-tools is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    anim-studio-tools is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with anim-studio-tools.  If not, see <http://www.gnu.org/licenses/>.
***/
