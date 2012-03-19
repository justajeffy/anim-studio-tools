/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: http://svn/drd/apps/bee/trunk/lib/gl/streams.h $"
 * SVN_META_ID = "$Id: streams.h 27186 2010-04-07 01:35:04Z david.morris $"
 */

//-------------------------------------------------------------------------------------------------
#ifndef bee_streams_H
#define bee_streams_H

#include "../kernel/types.h"

//-------------------------------------------------------------------------------------------------
namespace bee
{
//-------------------------------------------------------------------------------------------------
	class Stream
	{
	public:
		inline void setGLId( UInt a_glId )
		{
			m_glId = a_glId;
		}
		inline UInt getGLId() const
		{
			return m_glId;
		}
		inline UInt getGLComponentType() const
		{
			return m_GLComponentType;
		}

		inline UInt getNumElements() const
		{
			return m_NumElements;
		}
		inline UInt getNumComponentsPerElement() const
		{
			return m_NumComponentsPerElement;
		}

		inline UInt getComponentSize() const
		{
			return m_ComponentSize;
		}
		inline UInt getStride() const
		{
			return m_Stride;
		}
		inline UInt getSize() const
		{
			return m_Size;
		}

		inline const void * getData() const
		{
			return m_Data;
		}
		inline void * getData()
		{
			return m_Data;
		}

	protected:
		Stream( UInt a_GLComponentType,
				UInt a_NumElements,
				UInt a_NumComponentsPerElement,
				UInt a_ComponentSize )
			:	m_glId( 0 )
			,	m_GLComponentType( a_GLComponentType )
			,	m_NumElements( a_NumElements )
			,	m_NumComponentsPerElement( a_NumComponentsPerElement )
			,	m_ComponentSize( a_ComponentSize )
			,	m_Stride( a_NumComponentsPerElement * a_ComponentSize )
			,	m_Size( roundUp( a_NumElements * a_NumComponentsPerElement * a_ComponentSize, 16 ) )
			,	m_Data( NULL )
		{
			// you must delete in the (derived) destructor
			m_Data = new UChar [ m_Size ];
		}

		UInt m_glId;
		UInt m_GLComponentType;
		UInt m_NumElements;
		UInt m_NumComponentsPerElement;
		UInt m_ComponentSize;
		UInt m_Stride;
		UInt m_Size;

		UChar * m_Data;
	};

//-------------------------------------------------------------------------------------------------
	class VertexStream : public Stream
	{
	public:
		VertexStream( const char * a_Name,
					  UInt a_NumElements,
					  UInt a_NumComponentsPerElement,
					  UInt a_GLComponentType = GL_FLOAT, // 0x1406
					  UInt a_ComponentSize = sizeof( Float ) )
			:	Stream( a_GLComponentType, a_NumElements, a_NumComponentsPerElement, a_ComponentSize )
			,	m_Name( a_Name )
			{}
		~VertexStream() {	delete [] m_Data;	}

		inline const char * getName() const { return m_Name; }
	private:
		const char * m_Name;
	};

//-------------------------------------------------------------------------------------------------
	class IndexStream : public Stream
	{
	public:
		IndexStream( UInt a_NumElements,
					 UInt a_NumComponentsPerElement ,
					 UInt a_GLDrawType = GL_TRIANGLES, // 0x0004
					 UInt a_GLComponentType = GL_UNSIGNED_INT, // 0x1405
					 UInt a_ComponentSize = sizeof( UInt ) )
			:	Stream( a_GLComponentType, a_NumElements, a_NumComponentsPerElement, a_ComponentSize )
			,	m_GLDrawType( a_GLDrawType )
			{}
		~IndexStream() {	delete [] m_Data;	}

		inline UInt getGLDrawType() const { return m_GLDrawType; }
	private:
		UInt m_GLDrawType;
	};
}


#endif // bee_Streams_H


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
