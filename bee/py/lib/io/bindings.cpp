

#include <boost/python.hpp>
#include <boost/python/list.hpp>
#include <boost/python/dict.hpp>
#include <boost/python/extract.hpp>

#include <boost/pointer_cast.hpp>
#include <string>
#include <drdDebug/log.h>

#include <kernel/pythonHelper.h>
#include <io/Loader.h>
#include <io/MeshLoader.h>
#include <io/VacLoader.h>
#include <gl/Mesh.h>
#include <io/textureLoader.h>
#include <gl/Program.h>

//----------------------------------------------------------------------------
using namespace bee;
DRD_MKLOGGER( L, "drd.bee.io.bindings" );

//----------------------------------------------------------------------------
boost::shared_ptr< MeshLoader > CreateMeshLoader( const std::string & a_FileName )
{
	return Loader::CreateMeshLoader( a_FileName );
}

//----------------------------------------------------------------------------
boost::shared_ptr< VacLoader > CreateVacLoader( const std::string & a_FileName )
{
	boost::shared_ptr< MeshLoader > loader = Loader::CreateMeshLoader( a_FileName );
	DRD_LOG_ASSERT( L, loader->getType() == Loader::eVac, "Loader did not return a vacLoader!" );
	boost::shared_ptr< VacLoader > vacLoader ( boost::dynamic_pointer_cast< VacLoader >( loader.get() ) );
	return vacLoader;
}

//----------------------------------------------------------------------------
void importIOBindings()
{
	using namespace boost::python;
/*
	def( "CreateMeshLoader", CreateMeshLoader );
	class_< MeshLoader >( "MeshLoader", no_init )
		.def( "reportStats", pure_virtual( &MeshLoader::reportStats ) )
		.def( "open", pure_virtual( &MeshLoader::open ) )
		.def( "load", pure_virtual( &MeshLoader::load  ) )
		.def( "write", pure_virtual( &MeshLoader::write ) )
	;
*/

	def( "CreateVacLoader", CreateVacLoader );
	class_< VacLoader >( "VacLoader", no_init )
		.def( "reportStats", &VacLoader::reportStats )
		.def( "open", &VacLoader::open )
		.def( "load", &VacLoader::load  )
		.def( "write", &VacLoader::write )
	;

	class_<TextureLoader>( "TextureLoader", init< std::string >() )
		DEF_PY_FUNCTION( TextureLoader, unload )
		DEF_PY_FUNCTION( TextureLoader, createTexture )
		DEF_PY_FUNCTION( TextureLoader, reportStats )
	;
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
