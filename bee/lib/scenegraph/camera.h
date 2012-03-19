/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: http://svn/drd/apps/bee/trunk/lib/scenegraph/camera.h $"
 * SVN_META_ID = "$Id: camera.h 50498 2010-10-20 22:28:47Z stephane.bertout $"
 */

#ifndef bee_camera_h
#define bee_camera_h
#pragma once

#include "../kernel/string.h"
#include "../kernel/classHelper.h"
#include "../math/Imath.h"

namespace bee
{
	class Camera;
	typedef void (*fpCameraManagementCallBack)( const Camera * a_Light, bool a_Add );

	//-------------------------------------------------------------------------------------------------
	//! Camera is a Scenegraph utility class used to manipulate Camera
	class Camera
	{
	public:
		//! Constructor
		Camera( std::string a_Name );
		//! Destructor
		~Camera();

		//! Callback function called each time a Camera is created or deleted
		static void setCameraManagementCallBack( fpCameraManagementCallBack a_CameraManagementCallBack );

	private:
		static fpCameraManagementCallBack s_CameraManagementCallBack;

		ADD_MEMBER( std::string, Name )
		ADD_MEMBER_RDONLY( int, ID )
		ADD_MEMBER( int, ResolutionWidth )
		ADD_MEMBER( int, ResolutionHeight )
		ADD_MEMBER( float, FOV )
		ADD_MEMBER( float, NearClipPlane )
		ADD_MEMBER( float, FarClipPlane )

		ADD_MEMBER_RDONLY( Matrix, Transform )
		void setTransform( const Imath::V3f & a_RotX, const Imath::V3f & a_RotY, const Imath::V3f & a_RotZ, const Imath::V3f & a_Tr );
	};
}

#endif // bee_camera_h


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
