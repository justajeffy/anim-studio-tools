/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: http://svn/drd/apps/bee/trunk/lib/gl/Primitive2D.h $"
 * SVN_META_ID = "$Id: Primitive2D.h 27186 2010-04-07 01:35:04Z david.morris $"
 */

#ifndef bee_Primitive2D_h
#define bee_Primitive2D_h

#include "../kernel/types.h"

namespace bee
{
	class Program;

	//-------------------------------------------------------------------------------------------------
	//! Primitive2D is a GL utility class allowing creation and rendering of a simple 2D primitives (Quad, Triangle)
	class Primitive2D
	{
	public:
		//! Type supported
		enum Type
		{
			eQuad,
			eTriangle,
		};

		//! Constructor
		Primitive2D( Type a_Type );
		//! Destructor
		~Primitive2D() {}

		//! Returns the vertex count (3 for a triangle | 4 for a quad)
		inline UInt getVertexCount() const
		{
			if ( m_Type == eTriangle ) return 3;
			else return 4;
		}
		//! Creates the gl buffers (3,2 and 4 floats respectively for position, texCoord and Color)
		// TODO: use 2 floats instead of 3 for position and char component instead of float for color
		void create( 	const void * a_PositionBuffer,
						const void * a_TexCoordBuffer,
						const void * a_ColorBuffer );

		//! Setup the GL State with the specified Program
		void use( const Program * a_Program ) const;
		//! GL Draw
		void draw();

	private:
		Type m_Type;
		UInt m_PositionBufferID;
		UInt m_TexCoordBufferID;
		UInt m_ColorBufferID;
	};
}

#endif // bee_Primitive2D_h


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
