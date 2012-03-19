/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: $"
 * SVN_META_ID = "$Id: context.cpp 92477 2011-07-21 22:48:36Z chris.cooper $"
 */

//-------------------------------------------------------------------------------------------------
#include "context.h"
#include "log.h"
#include "singleton_rebuildable.h"
#include "utils.h"

// for initializing and detecting GL
#include "GL/glx.h"
#include <bee/gl/glExtensions.h>

// for detecting gpu
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cutil_inline.h>
#include <cutil_gl_inline.h>

// for detecting rib context
#include <rx.h>
#include <RixInterfaces.h>
#include <vector>

// for getting eye position
#include <nvMath.h>

#include <iostream>
#include <stdexcept>

// OpenMP
#include <omp.h>

/* #define CONTEXTINFO_REPORT_THREAD_INFO */
DRD_MKLOGGER( L, "drd.grind.Context" );

#define MY_LOG_ALWAYS( L, msg ) { std::cout << msg << std::endl; }

//-------------------------------------------------------------------------------------------------
using namespace grind;
using namespace drd;


// singleton access
ContextInfo& ContextInfo::instance()
{
	return RebuildableSingleton<ContextInfo>::instance();
}


//-------------------------------------------------------------------------------------------------
inline bool dl_get_option(	const std::string& option_name,
							std::vector< RtFloat >& result )
{
	RxInfoType_t o_result_type;
	RtInt o_result_count;

	// let the user know if the shutter was found
	return ( 0 == RxOption( option_name.c_str(), &( result[ 0 ] ), sizeof(RtFloat) * result.size(), &o_result_type, &o_result_count ) );
}

//-------------------------------------------------------------------------------------------------
ContextInfo::ContextInfo()
{
#ifdef CONTEXTINFO_REPORT_THREAD_INFO
	m_ThreadId = ( unsigned int ) pthread_self();
	DRD_LOG_INFO( L, "** constructing a ContextInfo in thread " << ( unsigned int ) pthread_self() );
#endif
	// only initialize gl extensions if gl is present
	if ( hasOpenGL() )
	{
		initGLExtensions();
#ifndef __DEVICE_EMULATION__
#if CUDART_VERSION >= 3000
		cudaGLSetGLDevice( cutGetMaxGflopsDeviceId() );
#endif
#endif
	}

	m_HasGPU = false;

	// detect GPU
	int deviceCount = 0;
	if ( cudaGetDeviceCount( &deviceCount ) == cudaSuccess )
	{
		m_HasGPU = deviceCount > 0;

		// emulation mode on a cpu only machine will still report a device, so check the driver version as well
		if( deviceCount == 1 ){
			int driverVersion = 0;
			cudaDriverGetVersion(&driverVersion);
			m_HasGPU = driverVersion != 0;
		}
	}

	// detect RX
	std::vector< RtFloat > result( 1 );
	m_HasRX = dl_get_option( "Frame", result );
}

//-------------------------------------------------------------------------------------------------
ContextInfo::~ContextInfo()
{
#ifdef CONTEXTINFO_REPORT_THREAD_INFO
	DRD_LOG_INFO( L, "** destroying a ContextInfo in thread " << ( unsigned int ) pthread_self() << ": " );
	if( m_ThreadId == (unsigned int) pthread_self() ){
		DRD_LOG_INFO( L, "\tSUCCESS (same thread as created) :-)" );
	} else {
		DRD_LOG_INFO( L, "\tFAIL!  (was created in thread " << m_ThreadId << ")" ) ;
	}
#endif
}


//-------------------------------------------------------------------------------------------------
static unsigned long inKB( unsigned long bytes )
{
	return bytes / 1024;
}

//-------------------------------------------------------------------------------------------------
static unsigned long inMB( unsigned long bytes )
{
	return bytes / ( 1024 * 1024 );
}

//-------------------------------------------------------------------------------------------------
static void printStats( unsigned long free_mem,
						unsigned long total_mem )
{
	MY_LOG_ALWAYS( L, "     Free : " << free_mem  << " bytes (" << inMB( free_mem  ) << " MB)" );
	MY_LOG_ALWAYS( L, "     Total: " << total_mem << " bytes (" << inMB( total_mem ) << " MB)" );
	MY_LOG_ALWAYS( L, "     " << (100.0 * free_mem / (double) total_mem) << "% free" );
}

//-------------------------------------------------------------------------------------------------
void getDeviceMemInfo( 	int device,
                       	unsigned int & free_mem,
						unsigned int & total_mem )
{
	CUdevice dev;
	CUcontext ctx;
	CUresult res;

	// note, assumes cuInit has been called
	cuDeviceGet( &dev, device );
	cuCtxCreate( &ctx, 0, dev );

#if CUDART_VERSION < 3000
	res = cuMemGetInfo( &free_mem, &total_mem );
#else
	size_t a,b;
	res = cuMemGetInfo( &a, &b );
	free_mem = a;
	total_mem = b;
#endif

	if ( res != CUDA_SUCCESS ) DRD_LOG_ERROR( L, "cuMemGetInfo failed! (status = " << res << ")" );
	cuCtxDestroy( ctx );
}

//-------------------------------------------------------------------------------------------------
void dumpDeviceMemInfo()
{
	unsigned int free_mem, total_mem;
	int gpuCount, i;

	if( cuInit( 0 ) == CUDA_SUCCESS ){
		cuDeviceGetCount( &gpuCount );

		for ( i = 0; i < gpuCount ; i++ )
		{
			getDeviceMemInfo( i, free_mem, total_mem );
			MY_LOG_ALWAYS( L, "Device: " << i );
			printStats( free_mem, total_mem );
		}
	}
}

//-------------------------------------------------------------------------------------------------
unsigned int ContextInfo::getFreeMem( int a_Gpu )
{
//	if( cuInit( 0 ) != CUDA_SUCCESS )
//		return 0;

	int gpuCount = 0;
	cuDeviceGetCount( &gpuCount );
	if ( a_Gpu >= gpuCount )
		return 0;

	unsigned int freeMem=0, totalMem=0;
	getDeviceMemInfo( a_Gpu, freeMem, totalMem );
	return freeMem;
}

//-------------------------------------------------------------------------------------------------
unsigned int ContextInfo::getTotalMem( int a_Gpu )
{
//	if( cuInit( 0 ) != CUDA_SUCCESS )
//		return 0;

	int gpuCount = 0;
	cuDeviceGetCount( &gpuCount );
	if ( a_Gpu >= gpuCount )
		return 0;

	unsigned int freeMem=0, totalMem=0;
	getDeviceMemInfo( a_Gpu, freeMem, totalMem );
	return totalMem;
}


//-------------------------------------------------------------------------------------------------
float ContextInfo::getGpuMemAvailable( int a_Gpu )
{
	if( cuInit( 0 ) != CUDA_SUCCESS )
		return 0;

	int gpuCount = 0;
	cuDeviceGetCount( &gpuCount );
	if ( a_Gpu >= gpuCount )
		return 0;

	if ( a_Gpu >= gpuCount )
		return 0.0f;

	unsigned int freeMem=0, totalMem=0;
	getDeviceMemInfo( a_Gpu, freeMem, totalMem );
	return float(freeMem) / float(totalMem);
}

//-------------------------------------------------------------------------------------------------
void ContextInfo::dump() const
{
	MY_LOG_ALWAYS( L, "Grind Context:"
				<< "   Build Type=" << (hasEmulation() ? "CPU" : "GPU" )
				<< ",  GPU=" << ( m_HasGPU ? "YES" : "NO" )
				<< ",  OpenGL=" << ( hasOpenGL() ? "YES" : "NO" )
				<< ",  Renderman RX=" << ( m_HasRX ? "YES" : "NO" ) );

	if( !hasEmulation() ){
		DRD_LOG_DEBUG( L, "CUDART_VERSION: " << CUDART_VERSION );
#ifdef __SUPPORT_SM_10__
		MY_LOG_ALWAYS( L, "Support SM_10=YES ");
#else
		MY_LOG_ALWAYS( L, "Support SM_10=NO ");
#endif
	}

	if( hasOpenGL() ){
		DRD_LOG_DEBUG( L, "Direct Rendering: " << ( glXIsDirect( glXGetCurrentDisplay(), glXGetCurrentContext() )  ? "YES" : "NO" ) );
	}

	MY_LOG_ALWAYS( L, "Current thread id: " << ( unsigned long int ) pthread_self() );

	if ( m_HasGPU )
	{
		dumpDeviceMemInfo();
	}

#ifdef _OPENMP
	MY_LOG_ALWAYS( L, "OpenMP max threads: " << omp_get_max_threads() );
#endif
}

//-------------------------------------------------------------------------------------------------
bool ContextInfo::hasGPU() const
{
	return m_HasGPU;
}

//-------------------------------------------------------------------------------------------------
bool ContextInfo::hasEmulation() const
{
#ifdef __DEVICE_EMULATION__
	return true;
#else
	return false;
#endif
}

//-------------------------------------------------------------------------------------------------
bool ContextInfo::hasOpenGL() const
{
	return glXGetCurrentContext() != NULL;
}

//-------------------------------------------------------------------------------------------------
bool ContextInfo::hasRX() const
{
	return m_HasRX;
}

//-------------------------------------------------------------------------------------------------
Imath::V3f ContextInfo::eyePos() const
{
	if( hasOpenGL() )
	{
		nv::matrix4f model_view;
		nv::vec4f origin( 0, 0, 0, 1 );
		glGetFloatv( GL_MODELVIEW_MATRIX, (float*) model_view.get_value() );
		nv::matrix4f mv_inv = inverse( model_view );
		nv::vec4f view_pos = mv_inv * origin;
		return Imath::V3f( view_pos.x, view_pos.y, view_pos.z );
	}
	throw drd::RuntimeError( grindGetRiObjectName() + "eyePos not supported in the current context" );
}


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
