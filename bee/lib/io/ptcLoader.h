/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: http://svn.drd.int/drd/apps/bee/trunk/lib/io/objLoader.h $"
 * SVN_META_ID = "$Id: ptcLoader.h 107811 2011-10-15 06:12:38Z stephane.bertout $"
 */

#ifndef bee_ptcLoader_h
#define bee_ptcLoader_h

#include <map>
#include <vector>
#include "../math/Imath.h"
#include "../kernel/smartPointers.h"

namespace bee
{
	class Mesh;
	class PtcNode;

	//! a loader for point cloud files
	class PtcLoader
	{
	public:
		//! default constructor
		PtcLoader();

		//! default destructor
		virtual ~PtcLoader();

		//! open the file and read header info, throw an error if there's a problem
		void open( const std::string& i_FilePath, int i_Density = 100 );

		//! close the point cloud file
		void close();

		//! read in all points
		void read();

		//! read in i_PointCount points
		void read( unsigned int i_PointCount );

		//! access the point count
		unsigned long getPointCount() const { return m_PointCount; }

		//! access the user variable count
		int getUserVariableCount() { return m_PtcUserVars.size(); }

		//! access the user data size
		int getUserDataSize() { return m_UserDataSize; }

		//! access a user variable name by index
		std::string getUserVariableName( int idx ) { return m_PtcUserVars[idx].first; }

		//! access a user variable type by index
		const std::string & getUserVariableType( int idx ) { return m_PtcUserVars[idx].second; }

		//! access the data for a param
		const std::vector< half >& getParam( const std::string& i_ParamName ) const { return m_DataMap[ i_ParamName ]; }

		const std::vector< float >& getPositionVec() const { return m_PositionVec; }

		const BBox & getBoundingBox() const { return m_BoundingBox; }

		//! Returns the created Mesh (TODO: make it inherit from MeshLoader)
		/*virtual */ boost::shared_ptr< Mesh > createMesh();

		/*void fillVector( std::vector< float > & o_Vec, const std::string & a_Name ) const
		{
			o_Vec = m_DataMap[ a_Name ];
		}*/

		/*std::vector<float> & getVector( const std::string & a_Name )
		{
			if ( a_Name == "P" ) return m_PositionVec;
			return m_DataMap[ a_Name ];
		}*/

		const PtcNode * generatePtcTree( int split_count );

		const Vec3 & getBakeCamPosition() const { return m_BakeCamPosition; }
		const Vec3 & getBakeCamLookAt() const { return m_BakeCamLookAt; }
		const Vec3 & getBakeCamUp() const { return m_BakeCamUp; }
		int getBakeWidth() const { return m_BakeWidth; }
		int getBakeHeight() const { return m_BakeHeight; }
		float getBakeAspect() const { return m_BakeAspect; }

		const Matrix & getViewProjMatrix() const { return m_ViewProjMatrix; }
		const Matrix & getInvViewProjMatrix() const { return m_InvViewProjMatrix; }

	private:

		//! number of points in file
		unsigned long m_PointCount;

		//! number of vars in file
		int m_UserVarCount;

		//! number of floats in user data
		int m_UserDataSize;

		//! handle to 3delight point cloud
		void* m_PointCloudHandle;

		BBox m_BoundingBox;

		//! the position data storage, all as float arrays
		mutable std::vector< float > m_PositionVec;
		//! the various data storage, all as half arrays
		mutable std::map< std::string, std::vector< half > > m_DataMap;

		typedef std::pair<std::string, std::string> PtcUserVar; // name - type
		std::vector<PtcUserVar> m_PtcUserVars;

		std::string m_FilePath;

		Vec3 m_BakeCamPosition, m_BakeCamLookAt, m_BakeCamUp;
		int m_BakeWidth, m_BakeHeight;
		float m_BakeAspect;
		Matrix m_ViewProjMatrix, m_InvViewProjMatrix;

		int m_Density;
	};

} // bee


#endif /* bee_ptcLoader_h */


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
