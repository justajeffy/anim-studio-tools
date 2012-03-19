/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: http://svn/drd/apps/bee/trunk/py/lib/gl/PyTexture.cpp $"
 * SVN_META_ID = "$Id: PyTexture.cpp 17883 2009-11-30 03:36:10Z david.morris $"
 */

//----------------------------------------------------------------------------
// system includes
#include <stdexcept>
// bee includes
#include <io/textureLoader.h>
// bee::py includes
#include "PyTexture.h"
#include "PyProgram.h"

//----------------------------------------------------------------------------
using namespace bee::py;

//----------------------------------------------------------------------------
Texture::Texture()
:	m_rawTex( NULL )
{
}

//----------------------------------------------------------------------------
Texture::Texture( const bee::Texture * a_rawTexture )
:	m_rawTex( a_rawTexture )
{

}

//----------------------------------------------------------------------------
Texture::Texture( const std::string& a_path )
:	m_rawTex( NULL )
{
	read( a_path );
}

//----------------------------------------------------------------------------
Texture::~Texture()
{
}

//----------------------------------------------------------------------------
void Texture::read( const std::string& a_path )
{
	try
	{
		bee::TextureLoader loader( a_path );
		m_refCountTex = loader.createTexture();
		if ( m_refCountTex ) return;

	}
	catch ( ... )
	{
	}

	throw std::runtime_error( std::string( "error loading texture from" ) + a_path );
}

//----------------------------------------------------------------------------
void Texture::dumpGL( float /*a_LOD*/ )
{
#if 0
	if( !m_tex ) return;
	glEnable( GL_TEXTURE_2D );
	glBindTexture( GL_TEXTURE_2D, _tex->GetId() );

	glBegin( GL_QUADS );
	glTexCoord2d(0.0,0.0); glVertex3f(0.0,0.0,0.0);
	glTexCoord2d(1.0,0.0); glVertex3f(1.0,0.0,0.0);
	glTexCoord2d(1.0,1.0); glVertex3f(1.0,0.0,1.0);
	glTexCoord2d(0.0,1.0); glVertex3f(0.0,0.0,1.0);
	glEnd();

	m_tex->Use();

	const static GLfloat p[] =
	{	1,1,1, -1,1,1, -1,-1,1, 1,-1,1, // v0-v1-v2-v3
		1,1,1, 1,-1,1, 1,-1,-1, 1,1,-1, // v0-v3-v4-v5
		1,1,1, 1,1,-1, -1,1,-1, -1,1,1, // v0-v5-v6-v1
		-1,1,1, -1,1,-1, -1,-1,-1, -1,-1,1, // v1-v6-v7-v2
		-1,-1,-1, 1,-1,-1, 1,-1,1, -1,-1,1, // v7-v4-v3-v2
		1,-1,-1, -1,-1,-1, -1,1,-1, 1,1,-1}; // v4-v7-v6-v5


	glBegin( GL_QUADS );
	glTexCoord2d(0.0,0.0); glVertex3f(0.0,0.0,0.0);
	glTexCoord2d(1.0,0.0); glVertex3f(1.0,0.0,0.0);
	glTexCoord2d(1.0,1.0); glVertex3f(1.0,0.0,1.0);
	glTexCoord2d(0.0,1.0); glVertex3f(0.0,0.0,1.0);
	glEnd();
#endif
}

//----------------------------------------------------------------------------
void Texture::use( UInt a_idx, const Program& a_program, bool a_setUniformTexSize, int a_location ) const
{
	if ( m_rawTex )
		m_rawTex->use( a_idx, a_program.getProgram(), a_setUniformTexSize, a_location );
	else
		m_refCountTex->use( a_idx, a_program.getProgram(), a_setUniformTexSize, a_location );
}

//----------------------------------------------------------------------------
void Texture::release( UInt a_idx ) const
{
	if ( m_rawTex )
		m_rawTex->release( a_idx );
	else
		m_refCountTex->release( a_idx );
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
