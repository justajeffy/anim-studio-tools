/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: http://svn.drd.int/drd/apps/bee/trunk/lib/io/objLoader.h $"
 * SVN_META_ID = "$Id: dtexLoader.h 68416 2011-02-22 05:00:28Z stephane.bertout $"
 */

#ifndef bee_dtexLoader_h
#define bee_dtexLoader_h

#include <map>
#include <vector>
#include "../math/Imath.h"
#include "../kernel/smartPointers.h"
// 3delight
#include "dtex.h"

namespace bee
{
	class Mesh;

	//! a loader for point cloud files
	class DtexLoader
	{
	public:
		//! default constructor
		DtexLoader();

		//! default destructor
		virtual ~DtexLoader();

		//! open the file and read header info, throw an error if there's a problem
		void open( const std::string& i_FilePath );

		//! close the point cloud file
		void close();

		//! read in all points
		void read();

		//! access the point count
		unsigned int getPointCount() const { return m_PointCount; }

		const BBox & getBoundingBox() const { return m_BoundingBox; }

		//! Returns the created Mesh (TODO: make it inherit from MeshLoader)
		/*virtual */ boost::shared_ptr< Mesh > createMesh();

		const std::vector< Vec3 > & getData() const { return m_Data; }

		int getWidth() const { return m_Width; }
		int getHeight() const { return m_Height; }

	private:

		//! number of points in file
		unsigned int m_PointCount;

		//! handle to 3delight dtex stuff
		DtexFile * m_File;
		DtexImage * m_Image;

	    int m_Width, m_Height, m_ImageCount, m_ChannelCount, m_TileWidth, m_TileHeight;

		BBox m_BoundingBox;

		std::string m_FilePath;

		std::vector< Vec3 > m_Data;
	};

} // bee


#endif /* bee_dtexLoader_h */


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
