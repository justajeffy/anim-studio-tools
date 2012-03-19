/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: http://svn.drd.int/drd/apps/bee/trunk/lib/io/objLoader.h $"
 * SVN_META_ID = "$Id: ptcTreeBuilder.h 41740 2010-08-08 23:21:56Z allan.johns $"
 */

#ifndef bee_ptcTreeBuilder_h
#define bee_ptcTreeBuilder_h

#include <OpenEXR/ImathVec.h>
#include "../kernel/assert.h"
#include <vector>

#include "ptcNode.h"

namespace bee
{
	class PtcTree;

	class PtcTreeBuilder
	{
		PtcCell * m_PtcCells;
		int m_SplitCount;

		Imath::V3f m_SplitRange;
		Imath::V3f m_Min;
		Imath::V3f m_Max;

		static PtcTreeBuilder * s_Instance;

	public:
		PtcTreeBuilder( int a_SplitCount, const Imath::V3f & a_Min, const Imath::V3f & a_Max );
		~PtcTreeBuilder();

		static PtcTreeBuilder * Instance() { Assert( s_Instance != NULL ); return s_Instance; }

		const PtcTree * BuildTree( int a_PointCount, std::vector< float > & a_VertexVec, std::vector< float > & a_NormalVec, std::vector< float > & radiusVec );
		void FreeMemory();

		PtcCell * GetPtcCell( int x, int y, int z );
		bool IsPtcCellRangeEmpty( int startX, int startY, int startZ, int endX, int endY, int endZ );

		void GetPtcCellsIndexers( const Imath::V3f & position, int & xPos, int & yPos, int & zPos );

		const Imath::V3f & GetMin() const { return m_Min; }
		const Imath::V3f & GetMax() const { return m_Max; }
		const Imath::V3f & GetSplitRange() const { return m_SplitRange; }
	};

} //  bee

#endif /* bee_ptcTreeBuilder_h */



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
