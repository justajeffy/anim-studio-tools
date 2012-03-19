/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: $"
 * SVN_META_ID = "$Id: cuda_manager.cpp 45312 2010-09-09 06:11:02Z chris.cooper $"
 */

//-------------------------------------------------------------------------------------------------
#include <drdDebug/log.h>
DRD_MKLOGGER(L,"drd.grind.CudaManager");

#include "cuda_manager.h"
#include "log.h"

#include <pthread.h>

#include <cuda_runtime.h>
#include <cutil_inline.h>
#include <cutil_gl_inline.h>

//-------------------------------------------------------------------------------------------------
using namespace grind;

//-------------------------------------------------------------------------------------------------
static CudaManager s_CudaManager;

//-------------------------------------------------------------------------------------------------
CudaManager::CudaManager()
: m_Success(false)
{
	DRD_LOG_DEBUG( L, "constructing cuda manager" );
	initialize();
}

void CudaManager::initialize()
{
	DRD_LOG_DEBUG( L, "initializing Cuda manager from thread: " << ( unsigned int ) pthread_self() );

	// note from section 3.2 of cuda programming guide
	// 'there is no explicit initialization function for the runtime'

#ifndef __DEVICE_EMULATION__
	// make sure there's a cuda device
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	if( deviceCount < 1 ){
		DRD_LOG_ERROR( L, "can't find any cuda devices" );
		return;
	}
	cudaSetDevice( cutGetMaxGflopsDeviceId() );
#if CUDART_VERSION >= 3000
	// need to do this with
	cudaGLSetGLDevice( cutGetMaxGflopsDeviceId() );
#endif
#endif

	s_CudaManager.m_Success = true;
}

//-------------------------------------------------------------------------------------------------
CudaManager::~CudaManager()
{
	DRD_LOG_DEBUG( L, "destroying Cuda manager from thread: " << ( unsigned int ) pthread_self() );
}


//-------------------------------------------------------------------------------------------------
bool CudaManager::good()
{
	return m_Success;
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
