/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: http://svn.drd.int/drd/apps/bee/trunk/lib/io/objLoader.h $"
 * SVN_META_ID = "$Id: ptcCell.cpp 41740 2010-08-08 23:21:56Z allan.johns $"
 */

#include "io/ptcCell.h"
#include "io/ptcTree.h"

#include "math/Imath.h"

using namespace bee;

void PtcCell::Add( const Imath::V3f & pos, const Imath::V3f & nrl, float a_Radius )
{
	if ( m_Count == m_Allocated )
	{
		if ( m_Count == 0 )
		{
			m_Allocated = 16;

			m_Elements = new char[ m_Allocated * SIZEOF_ELEMENT ];
		}
		else
		{
			m_Allocated *= 2;

			char * newElements = new char[ m_Allocated * SIZEOF_ELEMENT ]; // alloc memory for

			memcpy( newElements, m_Elements, m_Count * SIZEOF_ELEMENT );

			delete [] m_Elements;
			m_Elements = newElements;
		}
	}

	*(( Imath::V3f * )(m_Elements + m_Count * SIZEOF_ELEMENT + POSITION_OFFSET )) = pos;
	*(( Imath::V3f * )(m_Elements + m_Count * SIZEOF_ELEMENT + NORMAL_OFFSET )) = nrl;
	*(( float * )(m_Elements + m_Count * SIZEOF_ELEMENT + RADIUS_OFFSET )) = a_Radius;
	m_Count ++;
}

void PtcCell::Repack()
{
	char * newElementBuffer = NULL;
	PtcTree::Instance()->Allocate( m_Count, newElementBuffer );

	assert( PtcTree::Instance()->GetElementSize() == SIZEOF_ELEMENT );
	assert( newElementBuffer != NULL );

	char * newElements = (char *) newElementBuffer;

	memcpy( newElements, m_Elements, m_Count * SIZEOF_ELEMENT );

	delete [] m_Elements;
	m_Elements = newElements;
	m_Allocated = 0; // we haven't allocated any memory, now we just use the mempool
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
