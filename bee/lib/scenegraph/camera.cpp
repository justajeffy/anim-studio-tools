/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: http://svn/drd/apps/bee/trunk/lib/scenegraph/camera.cpp $"
 * SVN_META_ID = "$Id: camera.cpp 50498 2010-10-20 22:28:47Z stephane.bertout $"
 */

#include "scenegraph/camera.h"
#include "kernel/assert.h"

#include <OpenEXR/ImathVec.h>

using namespace bee;

fpCameraManagementCallBack Camera::s_CameraManagementCallBack = NULL;

void Camera::setCameraManagementCallBack( fpCameraManagementCallBack a_CameraManagementCallBack )
{
	Assert( s_CameraManagementCallBack == NULL ); // let be sure we store the callback only once
	s_CameraManagementCallBack = a_CameraManagementCallBack;
}

int s_CameraUniqueID = 0;

Camera::Camera( std::string a_Name )
: m_Name( a_Name )
, m_ID( s_CameraUniqueID++ )
, m_ResolutionWidth( 0 )
, m_ResolutionHeight( 0 )
, m_FOV( 90.0f )
, m_NearClipPlane( 0.1f )
, m_FarClipPlane( 10000.0f )
{
	m_Transform.makeIdentity();

	if ( s_CameraManagementCallBack != NULL )
	{
		( *s_CameraManagementCallBack )( this, true ); // add
	}
}

Camera::~Camera()
{
	if ( s_CameraManagementCallBack != NULL )
	{
		( *s_CameraManagementCallBack )( this, true ); // add
	}
}

void Camera::setTransform( const Imath::V3f & a_RotX, const Imath::V3f & a_RotY, const Imath::V3f & a_RotZ, const Imath::V3f & a_Tr )
{
	m_Transform = Matrix( a_RotX.x, a_RotX.y, a_RotX.z, 0.f,
						  a_RotY.x, a_RotY.y, a_RotY.z, 0.f,
						  a_RotZ.x, a_RotZ.y, a_RotZ.z, 0.f,
						  a_Tr.x, a_Tr.y, a_Tr.z, 1.f
						  );
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
