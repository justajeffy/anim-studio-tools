
#include <drdDebug/log.h>

#include "io/Loader.h"
#include "io/ObjLoader.h"
#include "io/VacLoader.h"

//----------------------------------------------------------------------------
using namespace bee;
using namespace std;
DRD_MKLOGGER( L, "drd.bee.io.Loader" );

//----------------------------------------------------------------------------
namespace
{
	//----------------------------------------------------------------------------
	Loader::Type
	parseType( const std::string & a_FileName )
	{
		if ( ObjLoader::is( a_FileName ) )
			return Loader::eObj;
		if ( VacLoader::is( a_FileName ) )
			return Loader::eVac;
		return Loader::eNone;
	}
}

//----------------------------------------------------------------------------
// static
boost::shared_ptr< MeshLoader >
Loader::CreateMeshLoader( const std::string & a_FileName )
{
	switch ( parseType( a_FileName ) )
	{
		case Loader::eObj: return boost::shared_ptr< MeshLoader >( new ObjLoader( a_FileName ) );
		case Loader::eVac: return boost::shared_ptr< MeshLoader >( new VacLoader( a_FileName ) );
		default:
			throw std::runtime_error( "Unknown file format: " + a_FileName );
	}
}

//----------------------------------------------------------------------------
Loader::Loader( const std::string & a_BaseFilename, Type a_Type )
:	m_BaseFilename( a_BaseFilename )
,	m_Type( a_Type )
{
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
