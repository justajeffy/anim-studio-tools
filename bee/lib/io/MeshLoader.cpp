
#include "io/MeshLoader.h"
#include <drdDebug/log.h>

DRD_MKLOGGER(L,"drd.bee.io.MeshLoader");

//----------------------------------------------------------------------------
using namespace bee;
using namespace std;
using namespace drd;

//----------------------------------------------------------------------------
MeshLoader::MeshLoader( const std::string & a_FileName, Loader::Type a_Type )
:	Loader( a_FileName, a_Type )
{
}

//----------------------------------------------------------------------------
const std::vector< Imath::V3f > &
MeshLoader::getVertexVector() const
{
	return m_VertexVector;
}

//----------------------------------------------------------------------------
const std::vector< Imath::V3f > &
MeshLoader::getNormalVector() const
{
	return m_NormalVector;
}

//----------------------------------------------------------------------------
const std::vector< Imath::V2f > &
MeshLoader::getTexCoordVector() const
{
	return m_TexCoordVector;
}

//----------------------------------------------------------------------------
const std::vector< MeshPolygon > &
MeshLoader::getPolygonVector() const
{
	return m_PolygonVector;
}

//----------------------------------------------------------------------------
bool
MeshLoader::isAnimatable()
{
	return false;
}

//----------------------------------------------------------------------------
bool
MeshLoader::setFrame( float a_Frame )
{
	DRD_LOG_ASSERT( L, false, "Not implemented for this loader type." );
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
