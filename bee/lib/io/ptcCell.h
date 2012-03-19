/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: http://svn.drd.int/drd/apps/bee/trunk/lib/io/objLoader.h $"
 * SVN_META_ID = "$Id: ptcCell.h 41740 2010-08-08 23:21:56Z allan.johns $"
 */

#ifndef bee_ptcCell_h
#define bee_ptcCell_h

#include <OpenEXR/ImathVec.h>
#include "../kernel/assert.h"

#define POSITION_OFFSET ( 0 )
#define POSITION_SIZE ( sizeof( Imath::V3f ) )
#define NORMAL_OFFSET ( POSITION_OFFSET + POSITION_SIZE )
#define NORMAL_SIZE ( sizeof( Imath::V3f ) )
#define RADIUS_OFFSET ( NORMAL_OFFSET + NORMAL_SIZE )
#define RADIUS_SIZE ( sizeof( float ) )

#define OUTPUT_SIZE ( sizeof( float ) ) // just the occlusion for now
#define OUTPUT_OFFSET ( RADIUS_OFFSET + RADIUS_SIZE )

#define SIZEOF_ELEMENT ( POSITION_SIZE + NORMAL_SIZE + RADIUS_SIZE + OUTPUT_SIZE )

/* position - normal - radius - output */

namespace bee
{
	class PtcCell
	{

		Imath::V3f 	m_Min,
					m_Max;
		int 	m_Count,
				m_Allocated;

		union
		{
			char * 	m_Elements;
			long int 	m_ElementsIdx;
		};

		long int m_NodeIdx;

	public:

		PtcCell()
		: m_Count( 0 )
		, m_Allocated( 0 )
		, m_Elements( NULL )
		, m_NodeIdx( -1 )
		{}

		void SetNodeIdx( long int a_NodeIdx )
		{
			Assert( m_NodeIdx == -1 );
			m_NodeIdx = a_NodeIdx;
		}
		long int GetNodeIdx() const { return m_NodeIdx; }

		// on purpose there's no destructor..

		const Imath::V3f & GetMin() const { return m_Min; }
		const Imath::V3f & GetMax() const { return m_Max; }
		int GetCount() const { return m_Count; }

		const char * GetElement( int idx ) const
		{
			Assert( idx < m_Count );
			return m_Elements + ( idx * SIZEOF_ELEMENT );
		}
		long int GetElementsIdx() const
		{
			return m_ElementsIdx;
		}

		const Imath::V3f & GetPosition( int idx ) const // only work when we deal with pointer (m_Elements instead of m_ElementsIdx)
		{
			return *( (const Imath::V3f * ) GetElement( idx ) );
		}
		const Imath::V3f & GetNormal( int idx ) const // only work when we deal with pointer (m_Elements instead of m_ElementsIdx)
		{
			return *(( (const Imath::V3f * ) GetElement( idx ) ) + 1 );
		}

		void SetBox( int xPos, int yPos, int zPos, const Imath::V3f & a_SplitRange, const Imath::V3f & a_Min )
		{
			if ( m_Count == 0 )
			{
				m_Min = a_Min + Imath::V3f(xPos, yPos, zPos) * a_SplitRange;
				m_Max = a_Min + Imath::V3f(xPos+1, yPos+1, zPos+1) * a_SplitRange;
			}
		}

		void Add( const Imath::V3f & pos, const Imath::V3f & nrl, float a_Radius );
		void Repack();

		void SetElementsIdx( const char * baseElements )
		{
			m_ElementsIdx = m_Elements - baseElements;
		}
};

} //  bee

#endif /* bee_ptcCell_h */



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
