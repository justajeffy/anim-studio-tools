/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: http://svn.drd.int/drd/apps/bee/trunk/lib/io/objLoader.h $"
 * SVN_META_ID = "$Id: ptcTree.h 41740 2010-08-08 23:21:56Z allan.johns $"
 */

#ifndef bee_ptcTree_h
#define bee_ptcTree_h

#include <OpenEXR/ImathVec.h>
#include "../kernel/assert.h"

#include "ptcNode.h"

namespace bee
{
	class PtcTree // mempool style
	{
	public: // dbg !
		int 	m_ElementCount,
				m_ElementAllocated,
				m_ElementSize;
		char * 	m_ElementBuffer;

		PtcCell * 	m_PtcCellBuffer;
		int 		m_PtcCellCount,
					m_PtcCellAllocated;

		PtcNode * 	m_PtcNodeBuffer;
		int 		m_PtcNodeCount,
					m_PtcNodeAllocated;

		static PtcTree * s_Instance;

	public:
		PtcTree( int a, int s );
		PtcTree( const std::string & a_FileName );
		PtcTree( const PtcTree & a, void * a_ElementBuffer, void * a_PtcCellBuffer, void * a_PtcNodeBuffer );
		~PtcTree();

		static PtcTree * Instance() { Assert( s_Instance != NULL ); return s_Instance; }

		void MoveAllocatedBuffer(); // dbg

		void Allocate( int count_needed, char * & o_ElementBuffer );

		bool IsEmpty() const
		{
			return ( m_ElementCount == m_ElementAllocated );
		}

		int GetElementCount() const
		{
			return m_ElementCount;
		}

		const char * GetElementBuffer() const
		{
			return m_ElementBuffer;
		}
		int GetElementBufferSize() const
		{
			return m_ElementAllocated * m_ElementSize;
		}
		int GetElementSize() const
		{
			return m_ElementSize;
		}

		int GetPtcCellCount() const { return m_PtcCellCount; }
		int GetPtcCellAllocated() const { return m_PtcCellAllocated; }
		int GetPtcNodeCount() const { return m_PtcNodeCount; }
		int GetPtcNodeAllocated() const { return m_PtcNodeAllocated; }

		const PtcCell * GetPtcCellBuffer() const { return m_PtcCellBuffer; }
		const PtcCell * GetPtcCell( int idx ) const
		{
			Assert( idx < m_PtcCellCount );
			return m_PtcCellBuffer + idx;
		}
		int GetPtcCellBufferSize() const { return sizeof(PtcCell) * m_PtcCellAllocated; }

		const PtcNode * GetPtcNodeBuffer() const { return m_PtcNodeBuffer; }
		const PtcNode * GetPtcNode( int idx ) const
		{
			Assert( idx < m_PtcNodeCount );
			return m_PtcNodeBuffer + idx;
		}
		int GetPtcNodeBufferSize() const { return sizeof(PtcNode) * m_PtcNodeAllocated; }

		const PtcNode * GetRootPtcNode() const
		{
			return GetPtcNode( 0 );
		}

		void ClearPtcCells()
		{
			delete [] m_PtcCellBuffer;
			m_PtcCellBuffer = NULL;
			m_PtcCellCount = m_PtcCellAllocated = 0;
		}

		void ClearPtcNodes()
		{
			delete [] m_PtcNodeBuffer;
			m_PtcNodeBuffer = NULL;
			m_PtcNodeCount = m_PtcNodeAllocated = 0;
		}

		void SaveToFile( const std::string & a_FileName );

		void AllocatePtcCellMemPool( int a )
		{
			Assert( m_PtcCellBuffer == NULL );
			m_PtcCellBuffer = new PtcCell [ a ];
			m_PtcCellAllocated = a;
		}
		void MovePtcCellMemPool() // debug
		{
			PtcCell * newPtcCells = new PtcCell [ m_PtcCellAllocated ];
			memcpy( newPtcCells, m_PtcCellBuffer, sizeof(PtcCell) * m_PtcCellAllocated );
			delete [] m_PtcCellBuffer;
			m_PtcCellBuffer = newPtcCells;
		}

		PtcCell * AllocatePtcCell()
		{
			Assert( m_PtcCellCount < m_PtcCellAllocated );
			return &m_PtcCellBuffer[ m_PtcCellCount ++ ];
		}

		bool IsPtcCellMemPoolEmpty()
		{
			return m_PtcCellCount == m_PtcCellAllocated;
		}

		void AllocatePtcNodeMemPool( int a )
		{
			assert( m_PtcNodeBuffer == NULL );
			m_PtcNodeBuffer = new PtcNode [ a ];
			m_PtcNodeAllocated = a;
		}
		void MovePtcNodeMemPool() // debug
		{
			PtcNode * newPtcNodes = new PtcNode [ m_PtcNodeAllocated ];
			memcpy( newPtcNodes, m_PtcNodeBuffer, sizeof(PtcNode) * m_PtcNodeAllocated );
			delete [] m_PtcNodeBuffer;
			m_PtcNodeBuffer = newPtcNodes;
		}
		long int GetIndex( const PtcNode * a_Node ) const
		{
			return a_Node - m_PtcNodeBuffer;
		}

		PtcNode * AllocatePtcNode()
		{
			assert( m_PtcNodeCount < m_PtcNodeAllocated );
			return m_PtcNodeBuffer + ( m_PtcNodeCount ++ );
		}

		bool IsPtcNodeMemPoolEmpty()
		{
			return m_PtcNodeCount == m_PtcNodeAllocated;
		}

	private:
		void Init();
		void LoadFromFile( const std::string & a_FileName );
	};

} //  bee

#endif /* bee_ptcTree_h */



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
