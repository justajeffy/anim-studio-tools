/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: http://svn/drd/apps/bee/trunk/lib/gl/Frame.cpp $"
 * SVN_META_ID = "$Id: Frame.cpp 27186 2010-04-07 01:35:04Z david.morris $"
 */

#include "Frame.h"

#include "../math/Imath.h"
#include "../kernel/log.h"

using namespace bee;

Frame::Frame()
: m_Position( 0, 0, 0)
, m_LookAtDirection( 0, 0, -1 )
, m_LookAtPosDistance( 1.f )
, m_RotationAngles( 0, 0 )
, m_RotateSpeedFactor( 0.003f )
, m_TranslateSpeedFactor( 0.02f )
, m_TruckSpeedFactor( 0.05f )
, m_Tumbler( false )
{
	// call update with default values !
	dirty();
}

void Frame::translate( 	Int a_Dx,
						Int a_Dy,
						Int a_Dz )
{
	update();

	Float dx = (Float) a_Dx;
	Float dy = (Float) a_Dy;
	Float dz = (Float) a_Dz;

	Vec3 rightVec, upVec, forwardVec;
	getRightVector( rightVec );
	getUpVector( upVec );
	getForwardVector( forwardVec );

	//SPAM(forwardVec);

	// todo expose speed translate factor
	m_Position += rightVec * dx * m_TranslateSpeedFactor;
	m_Position += upVec * dy * m_TranslateSpeedFactor;
	m_Position += forwardVec * dz * m_TranslateSpeedFactor;

	dirty();
}

void Frame::rotate( Int a_Rx,
					Int a_Ry )
{
	Float rx = (Float) a_Rx;
	Float ry = (Float) a_Ry;

	m_RotationAngles.x += rx * m_RotateSpeedFactor;
	m_RotationAngles.y += ry * m_RotateSpeedFactor;
	// todo normalize angles?

	//SPAMt("B:", m_LookAtDirection);

	Vec3 toVector( 0, 0, -1 ), tmp;

	Matrix rotMat; // we dont want z axis rotation
	rotMat.setEulerAngles( Vec3( m_RotationAngles.x, m_RotationAngles.y, 0.f ) );

	rotMat.multDirMatrix( toVector, m_LookAtDirection );

	/*Matrix rotXMat; rotXMat.setAxisAngle( Vec3(1,0,0), m_RotationAngles.x );
	 Matrix rotYMat; rotYMat.setAxisAngle( Vec3(0,1,0), m_RotationAngles.y );

	 rotXMat.multDirMatrix( toVector, tmp );
	 rotYMat.multDirMatrix( tmp, m_LookAtDirection );*/

	// SPAMt("A:", m_RotationAngles );

	dirty();
}

void Frame::truck( Int a_T )
{
	m_LookAtPosDistance -= float( a_T ) * m_TruckSpeedFactor;
	if ( m_LookAtPosDistance < 0.01f )
		m_LookAtPosDistance = 0.01f;
	dirty();
}

Vec3 Frame::getPosition() const
{
	if ( m_Tumbler )
		return m_Position + m_LookAtPosDistance * m_LookAtDirection;
	else
		return m_Position;
}


void Frame::update()
{
	if ( m_Tumbler )
		makeLookAt(m_Matrix, m_Position + m_LookAtPosDistance * m_LookAtDirection, m_Position, Vec3(0,1,0));
	else
		makeLookAt(m_Matrix, m_Position, m_Position + m_LookAtPosDistance * m_LookAtDirection, Vec3(0,1,0));

	/*
	LOG( DEBG, ( "d: %8.4f; p: %8.4f, %8.4f, %8.4f ; la: %8.4f, %8.4f, %8.4f",
			m_LookAtPosDistance,
			m_Position.x, m_Position.y, m_Position.z,
			m_LookAtDirection.x, m_LookAtDirection.y, m_LookAtDirection.z ) );
	*/
	//SPAM(m_Matrix);
	dirty( false );
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
