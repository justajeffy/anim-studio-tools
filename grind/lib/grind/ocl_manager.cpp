/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: $"
 * SVN_META_ID = "$Id: ocl_manager.cpp 45312 2010-09-09 06:11:02Z chris.cooper $"
 */

//-------------------------------------------------------------------------------------------------
#include "ocl_manager.h"
#include <pthread.h>
#include <drdDebug/log.h>

DRD_MKLOGGER( L, "drd.grind.OclManager" );

//-------------------------------------------------------------------------------------------------
grind::OclManager grind::s_OCLManager;

//-------------------------------------------------------------------------------------------------
grind::OclManager::OclManager()
{
}

void grind::OclManager::initialize()
{
#ifndef __DEVICE_EMULATION__
	DRD_LOG_DEBUG( L, "initializing OpenCL manager from thread: " << pthread_self() );

	cl_int status = CL_SUCCESS;

	grind::s_OCLManager.m_Context.reset( new cl::Context( CL_DEVICE_TYPE_GPU, 0, NULL, NULL, &status ) );
	grind::s_OCLManager.m_Devices = grind::s_OCLManager.m_Context->getInfo< CL_CONTEXT_DEVICES > ();
	DRD_LOG_DEBUG( L, " * found " << grind::s_OCLManager.m_Devices.size() << " OpenCL device(s) " );
	grind::s_OCLManager.m_Queue.reset( new cl::CommandQueue( *(grind::s_OCLManager.m_Context), grind::s_OCLManager.m_Devices[ 0 ], 0, &status ) );

	if ( status == CL_SUCCESS )
	{
		DRD_LOG_DEBUG( L, "success!" );
		return;
	}
	else
	{
		DRD_LOG_ERROR( L, "ERROR initializing OpenCL manager" );
	}
	grind::s_OCLManager.m_Context.reset( NULL );
#else
	DRD_LOG_INFO( L, "OclManager DISABLED in emulation mode" );
#endif
}

//-------------------------------------------------------------------------------------------------
grind::OclManager::~OclManager()
{
#ifndef __DEVICE_EMULATION__
	DRD_LOG_DEBUG( L, "destroying OpenCL manager from thread: " << pthread_self() );
#endif
}

//-------------------------------------------------------------------------------------------------
bool grind::OclManager::good()
{
	return m_Context != NULL;
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
