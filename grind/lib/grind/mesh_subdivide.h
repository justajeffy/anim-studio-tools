/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: $"
 * SVN_META_ID = "$Id: mesh_subdivide.h 42544 2010-08-17 04:31:03Z allan.johns $"
 */

#ifndef grind_mesh_subdivide_h
#define grind_mesh_subdivide_h

#include "mesh_subdivide_types.h"

#include "device_vector.h"
#include "vector_set.h"
#include "renderable.h"

#include "boost/shared_ptr.hpp"

namespace grind {

class DeviceMesh;
class GuideSet;

namespace subd {
struct IterationData;
}


//! catmull-clark mesh subdivision based on http://www.idav.ucdavis.edu/publications/print_pub?pub_id=964
struct MeshSubdivide
{
	//! default constructor
	MeshSubdivide();

	//! destructor
	~MeshSubdivide();

	//! subdivide a mesh
	void process( const DeviceMesh& i_Src, DeviceMesh& o_Dst );

	//! subdivide a guide set (mesh provides topology info)
	void processGuides( const DeviceMesh& i_SrcMesh, const GuideSet& i_Src, GuideSet& o_Dst );

	//! set the number of iterations
	void setIterations( int i_Val );

private:
	//! set up the per-iteration data
	void buildIterData();

	//! storage for the per-iteration data cache
	std::vector< boost::shared_ptr< subd::IterationData > > m_IterData;

	int m_Iterations, m_PrevIterations;
};

} // grind namespace

#endif /* grind_mesh_subdivide_h */


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
