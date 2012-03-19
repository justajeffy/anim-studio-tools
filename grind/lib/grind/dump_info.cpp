/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: $"
 * SVN_META_ID = "$Id: dump_info.cpp 49420 2010-10-14 01:57:33Z chris.cooper $"
 */

//-------------------------------------------------------------------------------------------------

#include "log.h"
DRD_MKLOGGER(L,"drd.grind.DumpInfo");

#define MY_LOG_ALWAYS( L, msg ) { std::cout << msg << std::endl; }

#include "dump_info.h"
#include "context.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

//-------------------------------------------------------------------------------------------------
void grind::dumpInfo()
{
	MY_LOG_ALWAYS( L, "Grind library compiled at " __TIME__ " on " __DATE__ );
	if( getenv( "GRIND_INSTALL_PATH" ) ){
		MY_LOG_ALWAYS( L, "$GRIND_INSTALL_PATH: " << getenv( "GRIND_INSTALL_PATH" ) );
	}
	grind::ContextInfo::instance().dump();
}

//-------------------------------------------------------------------------------------------------
void grind::dumpGPUInfo()
{
	MY_LOG_ALWAYS( L, "CUDA Device Query (from nvidia sdk)..." );

	int deviceCount = 0;

	if ( cudaGetDeviceCount( &deviceCount ) != cudaSuccess )
	{
		DRD_LOG_FATAL( L, "cudaGetDeviceCount failed! CUDA Driver and Runtime version may be mismatched." );
		DRD_LOG_FATAL( L, "Test FAILED!" );
		return;
	}

	// This function call returns 0 if there are no CUDA capable devices.
	if ( deviceCount == 0 )
		DRD_LOG_ERROR( L, "There is no device supporting CUDA" );

	int dev;
	for ( dev = 0; dev < deviceCount ; ++dev )
	{
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties( &deviceProp, dev );

		if ( dev == 0 )
		{
			// This function call returns 9999 for both major & minor fields, if no CUDA capable devices are present
			if ( deviceProp.major == 9999 && deviceProp.minor == 9999 )
			{
				DRD_LOG_ERROR( L, "There is no device supporting CUDA" );
			}
			else
			{
				MY_LOG_ALWAYS( L, "There are " << deviceCount << " device(s) supporting CUDA" );
			}
		}
		MY_LOG_ALWAYS( L, "Device " << dev << " '" << deviceProp.name << "'" );

#if CUDART_VERSION >= 2020
		int driverVersion = 0, runtimeVersion = 0;
		cudaDriverGetVersion(&driverVersion);
		MY_LOG_ALWAYS( L, "  CUDA Driver Version:                           " << driverVersion/1000 << "." << driverVersion%100);
		cudaRuntimeGetVersion(&runtimeVersion);
		MY_LOG_ALWAYS( L, "  CUDA Runtime Version:                          " << runtimeVersion/1000 << "." << runtimeVersion%100);
#endif

		MY_LOG_ALWAYS( L, "  CUDA Capability Major revision number:         " << deviceProp.major );
		MY_LOG_ALWAYS( L, "  CUDA Capability Minor revision number:         " << deviceProp.minor );

		MY_LOG_ALWAYS( L, "  Total amount of global memory:                 " << deviceProp.totalGlobalMem << " bytes" );
#if CUDART_VERSION >= 2000
		MY_LOG_ALWAYS( L, "  Number of multiprocessors:                     " << deviceProp.multiProcessorCount);
		MY_LOG_ALWAYS( L, "  Number of cores:                               " << ( 8 * deviceProp.multiProcessorCount ) );
#endif
		MY_LOG_ALWAYS( L, "  Total amount of constant memory:               " << deviceProp.totalConstMem << " bytes" );
		MY_LOG_ALWAYS( L, "  Total amount of shared memory per block:       " << deviceProp.sharedMemPerBlock << " bytes" );
		MY_LOG_ALWAYS( L, "  Total number of registers available per block: " << deviceProp.regsPerBlock );
		MY_LOG_ALWAYS( L, "  Warp size:                                     " << deviceProp.warpSize );
		MY_LOG_ALWAYS( L, "  Maximum number of threads per block:           " << deviceProp.maxThreadsPerBlock );
		MY_LOG_ALWAYS( L, "  Maximum sizes of each dimension of a block:    " << deviceProp.maxThreadsDim[ 0 ] << " x "
																		 << deviceProp.maxThreadsDim[ 1 ] << " x "
																		 << deviceProp.maxThreadsDim[ 2 ] );
		MY_LOG_ALWAYS( L, "  Maximum sizes of each dimension of a grid:     " << deviceProp.maxGridSize[ 0 ] << " x "
																		 << deviceProp.maxGridSize[ 1 ] << " x "
																		 << deviceProp.maxGridSize[ 2 ] );
		MY_LOG_ALWAYS( L, "  Maximum memory pitch:                          " << deviceProp.memPitch << " bytes" );
		MY_LOG_ALWAYS( L, "  Texture alignment:                             " << deviceProp.textureAlignment << " bytes" );
		MY_LOG_ALWAYS( L, "  Clock rate:                                    " << deviceProp.clockRate * 1e-6f  <<" GHz" );
#if CUDART_VERSION >= 2000
		MY_LOG_ALWAYS( L, "  Concurrent copy and execution:                 " << (deviceProp.deviceOverlap ? "Yes" : "No") );
#endif
#if CUDART_VERSION >= 2020
		MY_LOG_ALWAYS( L, "  Run time limit on kernels:                     " << ( deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No") );
		MY_LOG_ALWAYS( L, "  Integrated:                                    " << ( deviceProp.integrated ? "Yes" : "No") );
		MY_LOG_ALWAYS( L, "  Support host page-locked memory mapping:       " << ( deviceProp.canMapHostMemory ? "Yes" : "No") );
		MY_LOG_ALWAYS( L, "  Compute mode:                                  " << (deviceProp.computeMode == cudaComputeModeDefault ?
				"Default (multiple host threads can use this device simultaneously)" :
				deviceProp.computeMode == cudaComputeModeExclusive ?
				"Exclusive (only one host thread at a time can use this device)" :
				deviceProp.computeMode == cudaComputeModeProhibited ?
				"Prohibited (no host thread can use this device)" :
				"Unknown") );
#endif
	}
	MY_LOG_ALWAYS( L, "Test PASSED" );

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
