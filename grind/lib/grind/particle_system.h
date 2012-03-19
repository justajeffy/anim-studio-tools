/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: $"
 * SVN_META_ID = "$Id: particle_system.h 107813 2011-10-15 06:14:12Z stephane.bertout $"
 */

#ifndef grind_particle_system_h
#define grind_particle_system_h

#include "renderable.h"
#include "vector_set.h"
#include <string>
#include <map>
#include "bbox.h"
#include <bee/io/ptcLoader.h>

namespace grind {

//! a particle system
class ParticleSystem
: public Renderable
{
public:
	ParticleSystem();
	~ParticleSystem();

	//! return gpu mem usage of a file with a specific density (in percentage)
	unsigned long getGpuSizeOf( const std::string& i_Path, int i_Density = 100 );

	//! read from a file with a specific density (in percentage)
	void read( const std::string& i_Path, int i_Density = 100 );

	//! getBounds() from Renderable
	virtual BBox getBounds() const;

	//! get total point count
	int getPointCount() const { return m_PointCount; }

	//! get real point count (depending of the density)
	int getRealPointCount() const { return m_RealPointCount; }

	//! set UserData index to use (mainly for debugging purpose)
	void setCurrentUserDataIndex( int idx );

	//! get current UserData name
	std::string getCurrentUserDataName() const;

	//! get current UserData type
	std::string getCurrentUserDataType() const;

	//! get current UserData size
	int getCurrentUserDataSize() const;

	//! get UserData count
	int getUserDataCount() { return m_UserDataVecParams.size(); }

	//! get Width (dsm only)
	int getWidth() { return m_Width; }

	//! get Height (dsm only)
	int getHeight() { return m_Height; }

	//! get UserData name from index
	std::string getUserDataName( int idx) { return m_UserDataVecParams[idx].first; }

	//! get UserData type from index
	std::string getUserDataType( int idx) { return m_UserDataVecParams[idx].second; }

	//! static function to get UserData size from type
	static int getUserDataSizeFromType( std::string );

	// returning copy because of python..
	bee::Vec3 getBakeCamPosition() { return m_BakeCamPosition; }
	bee::Vec3 getBakeCamLookAt() { return m_BakeCamLookAt; }
	bee::Vec3 getBakeCamUp() { return m_BakeCamUp; }
	int getBakeWidth() const { return m_BakeWidth; }
	int getBakeHeight() const { return m_BakeHeight; }
	float getBakeAspect() const { return m_BakeAspect; }
	bee::Matrix getViewProjMatrix() const { return m_ViewProjMatrix; }
	bee::Matrix getInvViewProjMatrix() const { return m_InvViewProjMatrix; }

private:

	void dumpGL( float lod ) const;
	DeviceVectorSet m_Data;
	BBox m_Bounds;
	int m_PointCount, m_RealPointCount;
	bool m_PositionOnly;
	int m_Width, m_Height; // dsm only

	typedef std::pair<std::string, std::string> UserDataVecParam; // name - type
	std::vector<UserDataVecParam> m_UserDataVecParams;
	int m_CurrentUserDataIndex;

	bee::Vec3 m_BakeCamPosition, m_BakeCamLookAt, m_BakeCamUp;
	int m_BakeWidth, m_BakeHeight;
	float m_BakeAspect;
	bee::Matrix m_ViewProjMatrix, m_InvViewProjMatrix;
};

} // grind


#endif /* grind_particle_system_h */


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
