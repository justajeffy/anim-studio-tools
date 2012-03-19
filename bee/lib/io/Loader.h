
#ifndef bee_Loader_h
#define bee_Loader_h
#pragma once

#include "../kernel/string.h"
#include "../kernel/types.h"
#include "../kernel/smartPointers.h"

#include <string>
#include <vector>

namespace bee
{
	class MeshLoader;

	//----------------------------------------------------------------------------
	//! base loader class
	class Loader
	{
	public:
		enum Type
		{
			eObj,
			eVac,
			ePtc,
			eNone
		};

		static boost::shared_ptr< MeshLoader > CreateMeshLoader( const std::string & a_BaseFilename );

		virtual ~Loader() {}

		Type getType() const { return m_Type; }

		//! Stats report
		virtual void reportStats() = 0;

		virtual bool open() = 0;
		virtual bool load() = 0;
		virtual bool write() = 0;

	protected:
		Loader( const std::string & a_BaseFilename, Type a_Type );
		std::string m_BaseFilename;
		Type m_Type;
	};
}

#endif // bee_Loader_h


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
