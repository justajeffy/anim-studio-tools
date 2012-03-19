/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: http://svn.drd.int/drd/apps/bee/trunk/lib/io/objLoader.h $"
 * SVN_META_ID = "$Id: ptcNode.h 42896 2010-08-19 04:29:16Z stephane.bertout $"
 */

#ifndef bee_ptcNode_h
#define bee_ptcNode_h

#include <OpenEXR/ImathVec.h>
#include "../kernel/assert.h"

#include "ptcCell.h"

namespace bee
{

	class PtcNode
	{
		int m_StartX, m_StartY, m_StartZ;
		int m_EndX, m_EndY, m_EndZ;
		Imath::V3f m_Min, m_Max;

		union
		{
			union
			{
				PtcNode * m_Children;
				long int m_ChildrenIdx;
			};

			union
			{
				PtcCell * m_DataCell;
				long int m_DataCellIdx;
			};
		};

		union
		{
			PtcNode * m_Next;
			long int m_NextIdx;
		};

		union // can be the father as well
		{
			PtcNode * m_Prev;
			long int m_PrevIdx;
		};

		float padding[2];

	public:
		static int ptcNodeCreatedCount;

		PtcNode()
		: m_Children( NULL )
		, m_Next( NULL )
		, m_Prev( NULL )
		{
			++ptcNodeCreatedCount;
		}
		PtcNode( const Imath::V3f & a_Min, const Imath::V3f & a_Max, int sx, int sy, int sz, int ex, int ey, int ez )
		: m_StartX(sx), m_StartY(sy), m_StartZ(sz)
		, m_EndX(ex), m_EndY(ey), m_EndZ(ez)
		, m_Min( a_Min )
		, m_Max( a_Max )
		, m_Children( NULL )
		, m_Next( NULL )
		, m_Prev( NULL )
		{
			++ptcNodeCreatedCount;
		}
		~PtcNode()
		{
			--ptcNodeCreatedCount;
		}

		void operator = ( const PtcNode & s )
		{
			// copy everything except the pointers
			m_StartX = s.m_StartX; m_StartY = s.m_StartY; m_StartZ = s.m_StartZ;
			m_EndX = s.m_EndX; m_EndY = s.m_EndY; m_EndZ = s.m_EndZ;
			m_Min = s.m_Min; m_Max = s.m_Max;
		}

	    bool IsLeaf() const
	    {
	        return (m_EndX - m_StartX ) == 1;
	    }

		void AddNext( PtcNode * nextChild );
		void AddChild( PtcNode * newChild );

		int GetPointCount();
		const PtcNode * GetPrev() const { return m_Prev; }
		long int GetPrevIdx() const { return m_PrevIdx; }
		long int GetNextIdx() const { return m_NextIdx; }
		long int GetDataCellIdx() const { return m_DataCellIdx; }

		void Populate();

		// dbg stuff
		Imath::V3f CheckPositions() const;
		Imath::V3f CheckPositionsFromIndex() const;

		void RepackAssets();
		void RepackTo( PtcNode * newThis );
		void IndexPointers( const PtcNode * rootNode );
		void IndexDataCells( const PtcCell * baseCell );
		void IndexAssets( const char * baseAssets );

		typedef void (*fpcallFunctionOnDepthCallBack)( const PtcNode * a_Node );
		void callFunctionOnDepth( int depth, int current_depth, fpcallFunctionOnDepthCallBack fpCB ) const;

		static int add_done; // dbg a virer
	};
} //  bee

#endif /* bee_ptcNode_h */



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
