
#ifndef bee_MeshLoader_h
#define bee_MeshLoader_h
#pragma once

#include "../kernel/string.h"
#include "../kernel/types.h"
#include "../kernel/smartPointers.h"

#include "../io/Loader.h"

#include <OpenEXR/ImathVec.h>
#include <string>
#include <vector>
#include <boost/static_assert.hpp>

namespace bee
{
	//----------------------------------------------------------------------------
	class Mesh;

	//! Simple IO Utility Structure to store Material informations from an obj file
	class MeshMaterial
	{
	};

	//! Simple IO Utility Structure to store Face informations from an obj file
	struct MeshFace
	{
		Int vertexIdx, normalIdx, texCoordIdx;
		MeshFace( Int v, Int n, Int t ) : vertexIdx(v), normalIdx(n), texCoordIdx(t) {}
	};

	//! Simple IO Utility Structure to store Polygon informations from an obj file
	struct MeshPolygon
	{
		MeshPolygon( int count ) { m_FaceVector.reserve( count ); }
		std::vector< MeshFace > m_FaceVector;
	};

	BOOST_STATIC_ASSERT( sizeof( Imath::V3f ) == ( sizeof( float ) * 3 ) );
	BOOST_STATIC_ASSERT( sizeof( Imath::V2f ) == ( sizeof( float ) * 2 ) );
	BOOST_STATIC_ASSERT( sizeof( MeshFace ) == ( sizeof( int ) * 3 ) );

	//----------------------------------------------------------------------------
	class MeshLoader : public Loader
	{
	public:
		virtual ~MeshLoader() {}

		virtual void reportStats() = 0;

		virtual bool open() = 0;
		virtual bool load() = 0;
		virtual bool write() = 0;
		virtual bool close() = 0;

		//! Returns the created Mesh
		virtual boost::shared_ptr< Mesh > createMesh() = 0;

		virtual bool isAnimatable();
		virtual bool setFrame( float a_Frame );

		const std::vector< Imath::V3f > &   getVertexVector() const;
		const std::vector< Imath::V3f > &   getNormalVector() const;
		const std::vector< Imath::V2f > &   getTexCoordVector() const;
		const std::vector< MeshPolygon > &  getPolygonVector() const;

                virtual const std::vector< Imath::V3f > &   getStaticVertexVector() const   { return getVertexVector(); }
		virtual const std::vector< Imath::V3f > &   getStaticNormalVector() const   { return getNormalVector(); }
		virtual const std::vector< Imath::V2f > &   getStaticTexCoordVector() const { return getTexCoordVector(); }
		virtual const std::vector< MeshPolygon > &  getStaticPolygonVector() const  { return getPolygonVector(); }
	protected:
		MeshLoader( const std::string & a_FileName, Loader::Type a_Type );
		std::vector< Imath::V3f > 		m_VertexVector;
		std::vector< Imath::V3f > 		m_NormalVector;
		std::vector< Imath::V2f > 		m_TexCoordVector;
		std::vector< MeshPolygon >		m_PolygonVector;

	};
}

#endif // bee_MeshLoader_h


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
