/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: http://svn/drd/apps/bee/trunk/lib/gl/Frame.h $"
 * SVN_META_ID = "$Id: Frame.h 27186 2010-04-07 01:35:04Z david.morris $"
 */

#ifndef bee_Frame_h
#define bee_Frame_h
#pragma once

#include "../math/Imath.h"
#include "../kernel/types.h"
#include "../kernel/classHelper.h"

namespace bee
{
	//-------------------------------------------------------------------------------------------------
	//! Frame is a GL utility class allowing basic manipulation of a 3D Frame
	class Frame
	{
	public:
		//! Default constructor
		Frame();
		//! Destructor
		~Frame()
		{
		}

	// Getters
		//! Returns position as a Vec3
		Vec3 getPosition() const;
		//! Returns lookAt direction as a Vec3
		inline const Vec3 & getLookAtDirection() const
		{
			return m_LookAtDirection;
		}
		//! Returns distance between position and lookAt position (basically lookAtPos = position + lookAtDir * distance)
		inline Float getLookAtPosDistance() const
		{
			return m_LookAtPosDistance;
		}

		//! Returns the forward vector via a Vec3 reference (this assumes you have called GetMatrix or CheckDirty before)
		inline void getForwardVector( Vec3 & a_ForwardVector ) const
		{
			bee::getForwardVector( a_ForwardVector, m_Matrix );
		}
		//! Returns the up vector via a Vec3 reference (this assumes you have called GetMatrix or CheckDirty before)
		inline void getUpVector( Vec3 & a_UpVector ) const
		{
			bee::getRightVector( a_UpVector, m_Matrix );
		}
		//! Returns the right vector via a Vec3 reference (this assumes you have called GetMatrix or CheckDirty before)
		inline void getRightVector( Vec3 & a_RightVector ) const
		{
			bee::getUpVector( a_RightVector, m_Matrix );
		}

		//! Returns the matrix (will be recomputed if dirty)
		inline const Matrix & getMatrix()
		{
			checkDirty();
			return m_Matrix;
		}

	// Setters
		//! Set position via a Vec3
		void setPosition( const Vec3 & a_Pos )
		{
			m_Position = a_Pos;
			dirty();
		}
		//! Set lookAt direction via a Vec3
		void setLookAtDirection( const Vec3 & a_Dir )
		{
			m_LookAtDirection = a_Dir;
			dirty();
		}

	// Methods
		//! Method to translate the Frame of x|y|z units
		void translate( Int x, Int y, Int z );
		//! Method to rotate the Frame of x|y units
		void rotate( Int x, Int y );
		//! Method to truck the Frame of x|y units (will change the lookAtDistance and perform a zoom on the lookAt position)
		void truck( Int t );

		//! Update the matrix if dirty (something has changed)
		inline void checkDirty()
		{
			if ( m_Dirty ) update();
		}

		ADD_MEMBER( Float, RotateSpeedFactor );
		ADD_MEMBER( Float, TranslateSpeedFactor );
		ADD_MEMBER( Float, TruckSpeedFactor );
		ADD_MEMBER( Bool, Tumbler );

	private:
		void update();
		void dirty( Bool b = true )
		{
			m_Dirty = b;
		}

		Vec3 m_Position;
		Vec3 m_LookAtDirection;
		Float m_LookAtPosDistance;

		Bool m_Dirty;
		Matrix m_Matrix;

		Vec2 m_RotationAngles;
	};
}

#endif // bee_Frame_h


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
