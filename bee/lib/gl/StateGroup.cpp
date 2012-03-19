/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: http://svn/drd/apps/bee/trunk/lib/gl/StateGroup.cpp $"
 * SVN_META_ID = "$Id: StateGroup.cpp 30989 2010-05-11 23:23:13Z chris.cooper $"
 */

#include <GL/glew.h>
#include <GL/glut.h>
#include "StateGroup.h"
#include "../kernel/assert.h"
#include <algorithm>
#include <functional>
#include <iostream>

using namespace bee;
using namespace std;

//-----------------------------------------------------------------------------
void
StateGroup::State::use() const
{
	//printf( "\t: %d : 0x%04x : 0x%04x\n", m_Type, m_StateA, m_StateB );
	switch( m_Type )
	{
	case eEnable:		glEnable( m_StateA );								break;
	case eDisable:		glDisable( m_StateA );								break;
	// case eShadeModel:	glShadeModel( m_StateA );							break;
	case eBlendFunc:	glBlendFunc( m_StateA, m_StateB );					break;
	case eDepthFunc:	glDepthFunc( m_StateA );							break;
	case eDepthMask:	glDepthMask( m_StateA != 0 );						break;
	case eNoType:
		Assert( false && "Unknown type" );
	}
}

//-----------------------------------------------------------------------------
void
StateGroup::StateGroup::add( const State & a_State )
{
	m_States.push_back( a_State );
}

//-----------------------------------------------------------------------------
void
StateGroup::StateGroup::use() const
{
	for_each( m_States.begin(), m_States.end(), mem_fun_ref( &StateGroup::State::use ) );
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
