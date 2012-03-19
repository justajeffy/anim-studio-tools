/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: $"
 * SVN_META_ID = "$Id: texture.cpp 88551 2011-06-24 07:55:38Z luke.emrose $"
 */

//-------------------------------------------------------------------------------------------------
#include "texture.h"
#include "log.h"
#include "program.h"
#include "context.h"

#include "utils.h"

#include <drdDebug/log.h>
#include <drdDebug/runtimeError.h>

#include <GL/glew.h>
#include <GL/glut.h>

// bee includes
#include <bee/io/textureLoader.h>

#include <cutil_inline.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

#include <stdexcept>

DRD_MKLOGGER( L, "drd.grind.Texture" );

//-------------------------------------------------------------------------------------------------
using namespace grind;
using namespace drd;

//-------------------------------------------------------------------------------------------------
Texture::Texture()
{
	init();
}

//-------------------------------------------------------------------------------------------------
Texture::Texture( const std::string& path )
{
	init();
	read( path );
}

//-------------------------------------------------------------------------------------------------
void Texture::init()
{
	m_State = OPEN_GL;
	m_DevicePtr = NULL;
	m_Allocated = false;
}

//-------------------------------------------------------------------------------------------------
Texture::~Texture()
{}

//-------------------------------------------------------------------------------------------------
void Texture::read(const std::string& path)
{
	try {
		DRD_LOG_DEBUG( L, "Trying to load: " << path );
		bee::TextureLoader loader(path);
		m_Tex = loader.createTexture();
		return;
	} catch ( ... ) {}

	throw drd::RuntimeError( grindGetRiObjectName() + std::string("error loading texture from: ")+path );
}

//-------------------------------------------------------------------------------------------------
void Texture::dumpGL( float lod )
{
}

//-------------------------------------------------------------------------------------------------
void Texture::use( unsigned int idx, const Program& program, bool setUniformTexSize, int location ) const
{
	if( m_State != OPEN_GL ) prepForGL();

	m_Tex->use( idx, program.getProgram(), setUniformTexSize, location );
}

//-------------------------------------------------------------------------------------------------
void Texture::unUse( unsigned int idx ) const
{
	m_Tex->release( idx );
}

//-------------------------------------------------------------------------------------------------
void Texture::prepForCuda() const
{
	if( m_State == CUDA ) return;

	SAFE_CUDA( cudaGLMapBufferObject((void**)&m_DevicePtr, m_Tex->getId()) );
	m_State = CUDA;
}

//-------------------------------------------------------------------------------------------------
void Texture::prepForGL() const
{
	if( m_State == OPEN_GL ) return;

	SAFE_CUDA( cudaGLUnmapBufferObject( m_Tex->getId() ) );
	m_DevicePtr = NULL;
	m_State = OPEN_GL;
}

//-------------------------------------------------------------------------------------------------
unsigned int* Texture::getDevicePtr() const
{
	if( m_State != CUDA ) prepForCuda();
	return m_DevicePtr;
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
