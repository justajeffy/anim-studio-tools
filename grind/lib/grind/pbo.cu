/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: $"
 * SVN_META_ID = "$Id: $"
 */

//-------------------------------------------------------------------------------------------------
#include "cuda_types.h"
#include "log.h"
#include "pbo_types.h"
#include "timer.h"

#include <cutil_inline.h>
#include <cutil_math.h>

typedef unsigned int uint;
typedef unsigned char uchar;

DRD_MKLOGGER(L,"drd.grind.PBO");

// NOTE: texture references have to be declared at file scope! http://gpuray.blogspot.com/2009/07/texture-memory-all-abstraction-is-bad.html
//texture<float, 2> tex;
//texture<uchar4, 2, cudaReadModeNormalizedFloat> rgbaTex;
PBO_TEXTURE_REFERENCE_TYPE tex;

extern "C" void freeTextureDevice( CudaArrayHandle& i_Handle );

extern "C"
void setupTextureReference( PBO_TEXTURE_REFERENCE_TYPE* i_Tex )
{
	i_Tex->addressMode[ 0 ] = cudaAddressModeClamp;
	i_Tex->addressMode[ 1 ] = cudaAddressModeClamp;
	i_Tex->filterMode = cudaFilterModeLinear;
	i_Tex->normalized = true; // access with 0->1 coordinates
}

//-------------------------------------------------------------------------------------------------
extern "C"
void initTextureDevice(	int i_ImageWidth,
						int i_ImageHeight,
						const void * i_HostData,
						CudaArrayHandle& o_Handle )
{
	//! get rid of any existing data
	if( o_Handle.array != NULL ) freeTextureDevice( o_Handle );

	// allocate array and copy image data
	o_Handle.format = new cudaChannelFormatDesc;

	*o_Handle.format = cudaChannelFormatDesc( cudaCreateChannelDesc( 8, 8, 8, 8, cudaChannelFormatKindUnsigned ) );
	SAFE_CUDA( cudaMallocArray(&o_Handle.array, o_Handle.format, i_ImageWidth, i_ImageHeight) );
	uint size = i_ImageWidth * i_ImageHeight * sizeof(uchar) * 4;
	SAFE_CUDA( cudaMemcpyToArray(o_Handle.array, 0, 0, i_HostData, size, cudaMemcpyHostToDevice) );
	setupTextureReference( &tex );
	SAFE_CUDA( cudaBindTextureToArray( tex, o_Handle.array, *o_Handle.format ) );
}

//-------------------------------------------------------------------------------------------------
extern "C"
void pboBindDevice(	const CudaArrayHandle& i_Handle )
{
	SAFE_CUDA( cudaBindTextureToArray( tex, i_Handle.array, *i_Handle.format ) );
	//SAFE_CUDA( cudaBindTextureToArray( tex, i_Handle.array ) );
}

//-------------------------------------------------------------------------------------------------
extern "C"
void pboUnbindDevice()
{
	SAFE_CUDA( cudaUnbindTexture(tex) );
}


//-------------------------------------------------------------------------------------------------
extern "C"
void freeTextureDevice( CudaArrayHandle& i_Handle )
{
	if ( i_Handle.array != NULL )
	{
		SAFE_CUDA( cudaFreeArray( i_Handle.array ) );
		 i_Handle.array = NULL;
	}

	if( i_Handle.format != NULL )
	{
		delete i_Handle.format;
		 i_Handle.format = NULL;
	}
}


//-------------------------------------------------------------------------------------------------
// kernel that does the actual texture sampling
__global__
void sampleTextureKernel( 	unsigned int i_SampleCount,
							float* i_U,
							float* i_V,
							float* o_Result )
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if ( i >= i_SampleCount ) return;

	o_Result[ i ] = tex2D( tex, i_U[ i ], i_V[ i ] ).x;
}


//-------------------------------------------------------------------------------------------------
// entry point for texture sampling
extern "C"
void sampleTextureDevice( 	const CudaArrayHandle& i_Handle,
							unsigned int i_SampleCount,
							float* i_U,
							float* i_V,
							float* o_Result )
{
	pboBindDevice( i_Handle );

	dim3 block( GRID_DIM_X, GRID_DIM_Y, GRID_DIM_Y );
	dim3 grid( i_SampleCount / block.x + 1, 1, 1 );

	sampleTextureKernel<<<grid, block>>>( i_SampleCount, i_U, i_V, o_Result );

	pboUnbindDevice();
}
