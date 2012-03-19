
#ifndef bee_VacLoader_h
#define bee_VacLoader_h
#pragma once

#include <iostream>

#include "../kernel/string.h"
#include "../kernel/types.h"
#include "../kernel/smartPointers.h"
#include "../io/MeshLoader.h"

#include <OpenEXR/ImathVec.h>
#include <vacuum/VacDataAccessor.hpp>

namespace drd
{
	class VacDataAccessor;
}

namespace bee
{
	//----------------------------------------------------------------------------
	//! class for loading a vac mesh using bee
	class VacLoader : public MeshLoader
	{
	public:
		//----------------------------------------------------------------------------
        //! this will attempt to split the path for "easy" formatting
		//! returns true if the specified file path "is" a vac file, i.e.
		//! if it ends in a .h5 extension
		static bool is( const std::string & a_Filename );

		//----------------------------------------------------------------------------
		//!
		virtual void reportStats();

		//----------------------------------------------------------------------------
		//!
		virtual bool open();

		//----------------------------------------------------------------------------
		//!
		virtual bool load();

		//----------------------------------------------------------------------------
		//!
		virtual bool write();

		//----------------------------------------------------------------------------
		//!
		virtual bool close();

		//----------------------------------------------------------------------------
		//! Returns the created Mesh
		virtual boost::shared_ptr< Mesh > createMesh();

		//----------------------------------------------------------------------------
		//!
		virtual bool isAnimatable();

		//----------------------------------------------------------------------------
		//!
		virtual bool setFrame( float a_Frame );

		//----------------------------------------------------------------------------
		//!
        virtual const std::vector< Imath::V3f >&
        getStaticVertexVector() const
        {
        	return m_StaticVertexVector;
        }

		//----------------------------------------------------------------------------
		//!
		virtual const std::vector< Imath::V3f >&
		getStaticNormalVector() const
		{
			return m_StaticNormalVector;
		}

		//----------------------------------------------------------------------------
		//!
		virtual const std::vector< Imath::V2f >&
		getStaticTexCoordVector() const
		{
			return m_StaticTexCoordVector;
		}

		//----------------------------------------------------------------------------
		//!
		virtual const std::vector< MeshPolygon >&
		getStaticPolygonVector() const
		{
			return m_StaticPolygonVector;
		}


		//----------------------------------------------------------------------------
		//! set automatic application of transforms for vac files
		void setAutoApplyTransforms( bool const a_autoApplyTransforms );

		//----------------------------------------------------------------------------
		//! get the static of automatic application of transforms for vac files
		bool const getAutoApplyTransforms() const;

		//----------------------------------------------------------------------------
		//! get the reference frame to use for a vac file ( only of use for rendering )
		//! - specifies a frame that all other frames are transformed into, used for
		//! - the determination of local space
		void setReferenceFrame( float const a_frame );

	public:
		//----------------------------------------------------------------------------
		//! constructor takes a file path and generates a vac loader from it
		VacLoader( std::string const& a_CombinedPath );

		//----------------------------------------------------------------------------
		//! destructor
		virtual ~VacLoader();

	private:
		std::string m_Filename;
		std::string m_InternalPath;
		std::string m_OverridesFilename;
		std::string m_OverridesInternalPath;

		bool m_Loaded;
        bool m_HasOverrides;

		drd::VacDataAccessor * m_PropListAccessor;
		drd::VacDataAccessor * m_DeformListAccessor;

		std::vector< Imath::V3f > 		m_StaticVertexVector;
		std::vector< Imath::V3f > 		m_StaticNormalVector;
		std::vector< Imath::V2f > 		m_StaticTexCoordVector;
		std::vector< MeshPolygon >		m_StaticPolygonVector;

		bool m_autoApplyTransforms;
		float m_referenceFrame;
		float* m_referenceFrame_ptr;
	};
}

#endif // bee_VacLoader_h


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
