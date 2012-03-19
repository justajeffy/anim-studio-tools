/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: $"
 * SVN_META_ID = "$Id: cuda_gl_interop_wrapper.cpp 42544 2010-08-17 04:31:03Z allan.johns $"
 */

#include "cuda_gl_interop_wrapper.h"

#include <cuda_gl_interop.h>

//-------------------------------------------------------------------------------------------------
cudaError_t grind::cudaGLSetGLDevice(int device)
{
	return cudaGLSetGLDevice( device );
}


//-------------------------------------------------------------------------------------------------
cudaError_t grind::cudaGLRegisterBufferObject(MYuint bufObj)
{
	return cudaGLRegisterBufferObject( bufObj );
}


//-------------------------------------------------------------------------------------------------
cudaError_t grind::cudaGLMapBufferObject(void **devPtr, MYuint bufObj)
{
	return cudaGLMapBufferObject( devPtr, bufObj );
}


//-------------------------------------------------------------------------------------------------
cudaError_t grind::cudaGLUnmapBufferObject(MYuint bufObj)
{
	return cudaGLUnmapBufferObject( bufObj );
}


//-------------------------------------------------------------------------------------------------
cudaError_t grind::cudaGLUnregisterBufferObject(MYuint bufObj)
{
	return cudaGLUnregisterBufferObject( bufObj );
}


//-------------------------------------------------------------------------------------------------
cudaError_t grind::cudaGLSetBufferObjectMapFlags(MYuint bufObj, unsigned int flags)
{
	return cudaGLSetBufferObjectMapFlags( bufObj, flags );
}


//-------------------------------------------------------------------------------------------------
cudaError_t grind::cudaGLMapBufferObjectAsync(void **devPtr, MYuint bufObj, cudaStream_t stream)
{
	return cudaGLMapBufferObjectAsync( devPtr, bufObj, stream );
}


//-------------------------------------------------------------------------------------------------
cudaError_t grind::cudaGLUnmapBufferObjectAsync(MYuint bufObj, cudaStream_t stream)
{
	return cudaGLUnmapBufferObjectAsync( bufObj, stream );
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
