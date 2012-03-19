/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: http://svn/drd/apps/bee/trunk/lib/gl/Mesh.h $"
 * SVN_META_ID = "$Id: Mesh.h 30989 2010-05-11 23:23:13Z chris.cooper $"
 */

#ifndef bee_gl_Mesh_h
#define bee_gl_Mesh_h
#pragma once

#include <GL/glew.h>
#include <GL/glut.h>
#include "../kernel/types.h"

namespace bee
{
	class Program;

	//-------------------------------------------------------------------------------------------------
	//! Mesh is a GL utility class allowing creation and rendering of a Mesh (the streams supported are Vertex|Colour|Normal|TexCoord.. use GenericMesh if you need more streams..)
	class Mesh
	{
	public:
		//! Constructor taking the vertex data count to allocate (ex: 3 for a triangle)
		Mesh( UInt a_VertexDataCount );
		//! Destructor
		~Mesh();

		//! Create the vertex buffer using the parameters Buffer, Count and Size (ex: if you want a simple xyz position, Count=3 Size=sizeof(float))
		void createVertexBuffer( 	UInt Count,
									UInt Size,
									void * Buffer );
		//! Create the colour buffer using the parameters Buffer, Count and Size (ex: if you want a simple rgba int color, Count=4 Size=sizeof(char) Use sizeof(float) if you need a float for each component )
		void createColourBuffer( 	UInt Count,
									UInt Size,
									void * Buffer );
		//! Create the normal buffer using the parameters Buffer, Count and Size
		void createNormalBuffer( 	UInt Count,
									UInt Size,
									void * Buffer );
		//! Create the texture coordinate (UV0) buffer using the parameters Buffer, Count and Size (ex: if you want a simple uv texcoord, Count=2 Size=sizeof(float))
		void createTexCoordBuffer( 	UInt Count,
									UInt Size,
									void * Buffer );

		//! Setup the GL State with the specified Program
		void use( const Program * a_Program ) const;
		//! GL Draw using specific Draw Type
		// TODO: use enum instead of UInt...
		void draw( UInt a_Type = GL_TRIANGLES );

		//! Returns the vertex data count (constructor parameter)
		inline int getVertexDataCount()
		{
			return m_VertexDataCount;
		}

	private:

		inline UInt getVertexElementCount() const
		{
			return m_VertexElementCount;
		}
		inline UInt getVertexElementSize() const
		{
			return m_VertexElementSize;
		}
		inline UInt getColourElementCount() const
		{
			return m_ColourElementCount;
		}
		inline UInt getColourElementSize() const
		{
			return m_ColourElementSize;
		}
		inline UInt getNormalElementCount() const
		{
			return m_NormalElementCount;
		}
		inline UInt getNormalElementSize() const
		{
			return m_NormalElementSize;
		}
		inline UInt getTexCoordElementCount() const
		{
			return m_TexCoordElementCount;
		}
		inline UInt getTexCoordElementSize() const
		{
			return m_TexCoordElementSize;
		}

		UInt m_VertexBufferID;
		UInt m_ColourBufferID;
		UInt m_NormalBufferID;
		UInt m_TexCoordBufferID;

		UInt m_VertexDataCount;

		UInt m_VertexElementCount, m_VertexElementSize;
		UInt m_ColourElementCount, m_ColourElementSize;
		UInt m_NormalElementCount, m_NormalElementSize;
		UInt m_TexCoordElementCount, m_TexCoordElementSize;
	};
}

#endif // bee_gl_Mesh_h


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
