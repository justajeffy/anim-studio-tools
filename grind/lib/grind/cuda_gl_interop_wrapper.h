/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: $"
 * SVN_META_ID = "$Id: cuda_gl_interop_wrapper.h 42544 2010-08-17 04:31:03Z allan.johns $"
 */

#ifndef grind_cuda_gl_interop_wrapper_h
#define grind_cuda_gl_interop_wrapper_h

//! @cond DEV

//-------------------------------------------------------------------------------------------------

// for cudaError_t and cudaStream_t
#include <driver_types.h>

namespace grind {

//-------------------------------------------------------------------------------------------------
// note: this is a wrapper for cuda_gl_interop.h to remove it's inclusion of GL/gl.h
//       which isn't compatible with gl3.h

typedef unsigned int MYuint;

cudaError_t cudaGLSetGLDevice(int device);
cudaError_t cudaGLRegisterBufferObject(MYuint bufObj);
cudaError_t cudaGLMapBufferObject(void **devPtr, MYuint bufObj);
cudaError_t cudaGLUnmapBufferObject(MYuint bufObj);
cudaError_t cudaGLUnregisterBufferObject(MYuint bufObj);

cudaError_t cudaGLSetBufferObjectMapFlags(MYuint bufObj, unsigned int flags);
cudaError_t cudaGLMapBufferObjectAsync(void **devPtr, MYuint bufObj, cudaStream_t stream);
cudaError_t cudaGLUnmapBufferObjectAsync(MYuint bufObj, cudaStream_t stream);

} // grind

//! @endcond

#endif /* grind_cuda_gl_interop_wrapper_h */


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
