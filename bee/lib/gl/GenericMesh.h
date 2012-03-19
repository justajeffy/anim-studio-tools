/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: http://svn/drd/apps/bee/trunk/lib/gl/GenericMesh.h $"
 * SVN_META_ID = "$Id: GenericMesh.h 27186 2010-04-07 01:35:04Z david.morris $"
 */

#ifndef bee_GenericMesh_h
#define bee_GenericMesh_h
#pragma once

#include "../kernel/types.h"
#include "../math/Imath.h"

namespace bee
{
	class Program;

	class VertexStream;
	class IndexStream;

	//-------------------------------------------------------------------------------------------------
	//! GenericMesh is a GL utility class allowing creation and rendering of a Mesh said generic because any number of streams can be specified
	class GenericMesh
	{
	public:
		//! Constructor taking the vertex stream count to allocate and a boolean to precise if an index stream must be created or not
		GenericMesh( UInt a_NumberOfVertexStreams, Bool a_HasIndexStream );
		//! Destructor
		~GenericMesh();

		//! Function to add a vertex stream (this takes ownership of stream and returns an index) - Note to always add the vertex position first
		// that will determine the "vertex count"
		UInt addStream( VertexStream * a_Stream );
		//! Function to add a index stream
		Bool addStream( IndexStream * a_Stream );

		//! Setup the GL State with the specified Program
		void use( const Program * a_Program ) const;
		//! GL Draw
		void draw() const;
		//! GL Draw using specific Draw Type
		// TODO: use enum instead of UInt...
		void draw( UInt a_DrawType ) const;
		//! Unset the GL State (need to be called after the rendering to avoid side effects)
		void release( const Program * a_Program ) const;

		//! Returns the bouding box created from the position vertex stream
		const BBox & getBBox() const
		{
			return m_BBox;
		}

	private:
		VertexStream ** m_VertexStreams;
		IndexStream * m_IndexStream;

		BBox m_BBox;

		UInt m_NumberOfVertexStreams;
		Bool m_HasIndexStream;
	};

}

#endif // bee_GenericMesh_h


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
