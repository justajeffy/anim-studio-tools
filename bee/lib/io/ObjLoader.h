
#ifndef bee_ObjLoader_h
#define bee_ObjLoader_h
#pragma once

#include <iostream>

#include "../kernel/string.h"
#include "../kernel/types.h"
#include "../kernel/smartPointers.h"
#include "../io/MeshLoader.h"


namespace bee
{
	//----------------------------------------------------------------------------
	class Mesh;

	//----------------------------------------------------------------------------
	class ObjLoader : public MeshLoader
	{
	public:
		static bool is( const std::string & a_BaseFilename );

		virtual void reportStats();

		virtual bool open();
		virtual bool load();
		virtual bool write();
		virtual bool close();

		//! Returns the created Mesh
		virtual boost::shared_ptr< Mesh > createMesh();

		virtual bool setFrame( float a_Frame );

	public:
		ObjLoader( const std::string & a_BaseFilename );
		virtual ~ObjLoader() {}

	private:
		bool processLine( const char*& pBuffer);
		bool processLine( std::stringstream & is );
		UInt m_TotalFaceCount;
		int m_CurrentFrame;
		std::string m_CurrentFileName;
		bool m_Loaded;
	};
}

#endif // bee_ObjLoader_h


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
