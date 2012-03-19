/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: http://svn.drd.int/drd/apps/bee/trunk/lib/io/objLoader.h $"
 * SVN_META_ID = "$Id: ptcNode.cpp 41740 2010-08-08 23:21:56Z allan.johns $"
 */

#include "io/ptcNode.h"
#include "io/ptcTree.h"
#include "io/ptcTreeBuilder.h"

using namespace bee;

int PtcNode::ptcNodeCreatedCount = 0;

void PtcNode::AddNext( PtcNode * nextNode )
{
	if ( m_Next == NULL )
	{
		m_Next = nextNode;
	}
	else
	{
		m_Next->AddNext( nextNode );
	}
}

void PtcNode::AddChild( PtcNode * childNode )
{
	if ( m_Children == NULL )
	{
		m_Children = childNode;
	}
	else
	{
		m_Children->AddNext( childNode );
	}
}

int PtcNode::GetPointCount()
{
	int count = 0;

	if ((m_EndX - m_StartX) == 1)
	{
		count += m_DataCell->GetCount();
	}
	else
	{
		if (m_Children != NULL) count += m_Children->GetPointCount();
	}

	if (m_Next != NULL) count += m_Next->GetPointCount();

	return count;
}

void PtcNode::IndexPointers( const PtcNode * rootNode )
{
	if ((m_EndX - m_StartX) > 1)
	{
		if (m_Children != NULL)
		{
			m_Children->IndexPointers( rootNode );
			m_ChildrenIdx = m_Children - rootNode;
		}
		else m_ChildrenIdx = -1;
	}

	if (m_Next != NULL)
	{
		m_Next->IndexPointers( rootNode );
		m_NextIdx = m_Next - rootNode;
	}
	else m_NextIdx = -1;

	if ( m_Prev == NULL )
	{
		Assert( this == rootNode );
		m_PrevIdx = -1;
	}
	else
		m_PrevIdx = (((long int)m_Prev) - ((long int)rootNode)) / sizeof(PtcNode);
}

void PtcNode::IndexDataCells( const PtcCell * baseCell )
{
	if ((m_EndX - m_StartX) == 1)
	{
		m_DataCell->SetNodeIdx( PtcTree::Instance()->GetIndex( this ) );

		m_DataCellIdx = (((long int)m_DataCell) - ((long int)baseCell)) / sizeof(PtcCell);
	}
	else
	{
		if (m_Children != NULL) m_Children->IndexDataCells( baseCell );
	}

	if (m_Next != NULL) m_Next->IndexDataCells( baseCell );
}

void PtcNode::IndexAssets( const char * baseAssets )
{
	if ((m_EndX - m_StartX) == 1)
	{
		m_DataCell->SetElementsIdx( baseAssets );
	}
	else
	{
		if (m_Children != NULL) m_Children->IndexAssets( baseAssets );
	}

	if (m_Next != NULL) m_Next->IndexAssets( baseAssets );
}

void PtcNode::Populate()
{
	int depth = m_EndX - m_StartX;
	int half_depth = depth / 2;
	PtcTreeBuilder * treeBuilderInst = PtcTreeBuilder::Instance();

	if ( depth == 1 )
	{
		assert( m_DataCell == NULL );
		m_DataCell = treeBuilderInst->GetPtcCell( m_StartX, m_StartY, m_StartZ );
	}
	else
	{
		assert( m_Max != m_Min );
		Imath::V3f cellSize = ( m_Max - m_Min ) * 0.5f;

		for (int iX = 0; iX < 2; iX++ )
		{
			for (int iY = 0; iY < 2; iY++ )
			{
				for (int iZ = 0; iZ < 2; iZ++ )
				{
					int curStartX, curStartY, curStartZ, curEndX, curEndY, curEndZ;

					Imath::V3f curMin = m_Min + Imath::V3f( iX, iY, iZ ) * cellSize;
					Imath::V3f curMax = m_Min + Imath::V3f( iX+1, iY+1, iZ+1 ) * cellSize;

					curStartX = (iX == 0) ? (m_StartX) : (m_StartX + half_depth);
					curStartY = (iY == 0) ? (m_StartY) : (m_StartY + half_depth);
					curStartZ = (iZ == 0) ? (m_StartZ) : (m_StartZ + half_depth);
					curEndX = (iX == 0) ? (m_StartX + half_depth) : (m_EndX);
					curEndY = (iY == 0) ? (m_StartY + half_depth) : (m_EndY);
					curEndZ = (iZ == 0) ? (m_StartZ + half_depth) : (m_EndZ);

					depth = curEndX - curStartX;
					assert( depth == (curEndY - curStartY) );
					assert( depth == (curEndZ - curStartZ) );

					bool empty = treeBuilderInst->IsPtcCellRangeEmpty( curStartX, curStartY, curStartZ, curEndX, curEndY, curEndZ );
					if ( !empty )
					{
						PtcNode * newChild = new PtcNode( curMin, curMax, curStartX, curStartY, curStartZ, curEndX, curEndY, curEndZ );
						AddChild( newChild );
						newChild->Populate();
					}
				}
			}
		}
	}
}

int PtcNode::add_done = 0; // dbg
Imath::V3f PtcNode::CheckPositions() const
{
	const Imath::V3f & Min = PtcTreeBuilder::Instance()->GetMin();

	Imath::V3f diff( 0, 0, 0 );

	if ((m_EndX - m_StartX) == 1)
	{
		for( int i = 0; i < m_DataCell->GetCount(); ++i )
		{
			diff += ( m_DataCell->GetPosition( i ) - Min );
			add_done++;
		}
	}
	else
	{
		if (m_Children != NULL) diff += m_Children->CheckPositions();
	}

	if (m_Next != NULL) diff += m_Next->CheckPositions();

	return diff;
}

Imath::V3f PtcNode::CheckPositionsFromIndex() const
{
	const Imath::V3f & Min = PtcTreeBuilder::Instance()->GetMin();
	PtcTree * tree = PtcTree::Instance();

	Imath::V3f diff( 0, 0, 0 );

	if ((m_EndX - m_StartX) == 1)
	{
		const PtcCell * tdataCell = tree->GetPtcCell( m_DataCellIdx );
		const char * elements = ( tdataCell->GetElementsIdx() + tree->GetElementBuffer() );

		for( int i = 0; i < tdataCell->GetCount(); ++i )
		{
			const Imath::V3f * position = ( const Imath::V3f * ) ( elements + SIZEOF_ELEMENT * i + POSITION_OFFSET );

			diff += ( *position - Min );
			add_done++;
		}
	}
	else
	{
		if (m_ChildrenIdx != -1)
		{
			const PtcNode * tchildren = tree->GetPtcNode( m_ChildrenIdx );
			diff += tchildren->CheckPositionsFromIndex();
		}
	}

	if (m_NextIdx != -1)
	{
		const PtcNode * tnext = tree->GetPtcNode( m_NextIdx );
		diff += tnext->CheckPositionsFromIndex();
	}

	return diff;
}

void PtcNode::RepackAssets()
{
	if ((m_EndX - m_StartX) == 1)
	{
		m_DataCell->Repack();

		PtcCell * newCell = PtcTree::Instance()->AllocatePtcCell();
		*newCell = *m_DataCell; // copy content
		m_DataCell = newCell;
	}
	else
	{
		if (m_Children != NULL) m_Children->RepackAssets();
	}

	if (m_Next != NULL) m_Next->RepackAssets();
}

void PtcNode::RepackTo( PtcNode * newThis )
{
	(*newThis) = (*this);

	if ((m_EndX - m_StartX) == 1)
	{
		newThis->m_DataCell = m_DataCell;
	}
	else
	{
		if ( m_Children != NULL )
		{
			PtcNode * newChildren = PtcTree::Instance()->AllocatePtcNode();
			m_Children->RepackTo( newChildren );
			newThis->m_Children = newChildren;
			newChildren->m_Prev = newThis;

			delete m_Children;
		}
	}

	if (m_Next != NULL)
	{
		PtcNode * newNext = PtcTree::Instance()->AllocatePtcNode();
		m_Next->RepackTo( newNext );
		newThis->m_Next = newNext;
		newNext->m_Prev = newThis;

		delete m_Next;
	}
}

void PtcNode::callFunctionOnDepth( int depth_to_match, int current_depth, fpcallFunctionOnDepthCallBack fpCB ) const
{
	if (depth_to_match == current_depth)
	{
		(*fpCB)( this );
	}
	else if ((m_EndX - m_StartX) != 1)
	{
		if (m_Children != NULL) m_Children->callFunctionOnDepth(depth_to_match, current_depth + 1, fpCB );
	}

	if (m_Next != NULL) m_Next->callFunctionOnDepth(depth_to_match, current_depth, fpCB );
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
