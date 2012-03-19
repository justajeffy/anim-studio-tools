/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: http://svn/drd/apps/bee/trunk/lib/gl/StateGroup.h $"
 * SVN_META_ID = "$Id: StateGroup.h 27186 2010-04-07 01:35:04Z david.morris $"
 */

#ifndef bee_StateGroup_h
#define bee_StateGroup_h
#pragma once

#include "../kernel/types.h"
#include <vector>

//-------------------------------------------------------------------------------------------------
namespace bee
{
	class StateGroup
	{
	public:

//-------------------------------------------------------------------------------------------------
		class State
		{
		public:
			enum Type
			{
				eNoType,
				eEnable,
				eDisable,
				eShadeModel,
				eBlendFunc,
				eDepthFunc,
				eDepthMask,
			};

			State() :
				m_Type( eNoType )
			{
			}
			State( 	Type a_Type,
					UShort a_StateA ) :
				m_Type( a_Type ), m_StateA( a_StateA )
			{
			}
			State( 	Type a_Type,
					UShort a_StateA,
					UShort a_StateB ) :
				m_Type( a_Type ), m_StateA( a_StateA ), m_StateB( a_StateB )
			{
			}
			void use() const;

			Type m_Type;
			UShort m_StateA;
			UShort m_StateB;

		};

//-------------------------------------------------------------------------------------------------
		StateGroup()
		{
		}
		~StateGroup()
		{
		}

//-------------------------------------------------------------------------------------------------
		void add( const State & a_State );
		void use() const;

//-------------------------------------------------------------------------------------------------
	private:
		std::vector< State > m_States;
	};

}

#endif // bee_StateGroup_h


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
