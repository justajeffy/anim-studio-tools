/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: $"
 * SVN_META_ID = "$Id: particle_system.cpp 107813 2011-10-15 06:14:12Z stephane.bertout $"
 */

#include <drdDebug/log.h>
DRD_MKLOGGER(L,"drd.grind.ParticleSystem");

#include "particle_system.h"
#include "log.h"
#include "device_vector.h"
#include "host_vector.h"
#include "timer.h"
#include "cuda_types.h"

#define __gl_h_
#include <GL/glx.h> // FIX THAT !
#undef __gl_h_

#include <bee/io/dtexLoader.h>
#include <boost/filesystem.hpp>

using namespace grind;
using namespace drd;
using namespace boost::filesystem;
using namespace std;

//-------------------------------------------------------------------------------------------------
ParticleSystem::ParticleSystem()
: m_PointCount( 0 )
, m_RealPointCount( 0 )
, m_PositionOnly( false )
, m_Width( 0 )
, m_Height( 0 )
{
	DRD_LOG_INFO( L, "creating a particle system" );
}

//-------------------------------------------------------------------------------------------------
ParticleSystem::~ParticleSystem()
{
	DRD_LOG_INFO( L, "destroying a particle system" );
}

//-------------------------------------------------------------------------------------------------
BBox ParticleSystem::getBounds() const
{
	return m_Bounds;
}

//-------------------------------------------------------------------------------------------------
std::string ParticleSystem::getCurrentUserDataName() const
{
	if ( m_CurrentUserDataIndex == -1 ) return "None";
	return m_UserDataVecParams[m_CurrentUserDataIndex].first;
}

//-------------------------------------------------------------------------------------------------
std::string ParticleSystem::getCurrentUserDataType() const
{
	if ( m_CurrentUserDataIndex == -1 ) return "Unknown";
	return m_UserDataVecParams[m_CurrentUserDataIndex].second;
}

//-------------------------------------------------------------------------------------------------
int ParticleSystem::getUserDataSizeFromType( std::string type )
{
    	 if ( type == "color" )  	return 3;
	else if ( type == "point" )  	return 3;
	else if ( type == "normal" ) 	return 3;
	else if ( type == "vector" ) 	return 3;
	else if ( type == "float" )  	return 1;
	else 							return 0;
}

//-------------------------------------------------------------------------------------------------
int ParticleSystem::getCurrentUserDataSize() const
{
	if ( m_CurrentUserDataIndex == -1 ) return 0;
	return getUserDataSizeFromType( m_UserDataVecParams[m_CurrentUserDataIndex].second );
}

//-------------------------------------------------------------------------------------------------
void ParticleSystem::setCurrentUserDataIndex( int idx )
{
	m_CurrentUserDataIndex = -1;
	if ( m_UserDataVecParams.size() > idx )
		m_CurrentUserDataIndex = idx;
}

//-------------------------------------------------------------------------------------------------
unsigned long ParticleSystem::getGpuSizeOf( const std::string& i_Path, int i_Density )
{
	DRD_LOG_INFO( L, "get particle system gpu mem size of: " << i_Path.c_str() );

	path filePath( i_Path );
	std::string ext = filePath.extension();
	for(int i=0; i<ext.length(); ++i) ext[i] = std::tolower(ext[i]);

	if ( ext.compare( ".dsm" ) == 0 )
	{
		bee::DtexLoader loader;
		loader.open( i_Path );

		int pointCount = loader.getPointCount();
		return pointCount * sizeof(float) * 3;
	}
	else
	{
		bee::PtcLoader loader;
		loader.open( i_Path, i_Density );

		unsigned long pointCount = loader.getPointCount();
		int user_data_size = loader.getUserDataSize();
		unsigned long gpuMemSize = 0;

		gpuMemSize += pointCount * sizeof(float) * 3; // position
		gpuMemSize += pointCount * sizeof(Imath::V4h); // normal + radius
		gpuMemSize += pointCount * sizeof(half) * user_data_size; // user data as half

		return gpuMemSize;
	}
}

//-------------------------------------------------------------------------------------------------
void ParticleSystem::read( const std::string& i_Path, int i_Density )
{
	DRD_LOG_INFO( L, "reading particle system from: " << i_Path.c_str() );

	path filePath( i_Path );
	std::string ext = filePath.extension();
	for(int i=0; i<ext.length(); ++i) ext[i] = std::tolower(ext[i]);

	if ( ext.compare( ".dsm" ) == 0 )
	{
		CPUTimer t;
		t.start();
		bee::DtexLoader loader;
		loader.open( i_Path );
		loader.read();
		t.stop();
		DRD_LOG_INFO( L, "load time: " << t.elapsed_sec() << "s" );

		t.start();

		m_Width = loader.getWidth();
		m_Height = loader.getHeight();
		m_Bounds = BBox( loader.getBoundingBox().getMin(),
						 loader.getBoundingBox().getMax() );

		m_PointCount = loader.getPointCount();
		if ( m_PointCount > 0 )
		{
			m_RealPointCount = loader.getPointCount() * i_Density / 100;
			int point_step = 100 / i_Density;

			// get position ONLY
			m_PositionOnly = true;

			m_Data.getVec3fParam( "P" ).setBufferType( GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW );
			m_Data.getVec3fParam( "P" ).setValue( loader.getData() );

			t.stop();
			DRD_LOG_INFO( L, "pack/transfer time: " << t.elapsed_sec() << "s" );
		}
	}
	else
	{
		CPUTimer t;
		t.start();
		bee::PtcLoader loader;
		loader.open( i_Path, i_Density );
		loader.read();
		t.stop();
		DRD_LOG_INFO( L, "load time: " << t.elapsed_sec() << "s" );

		t.start();

		m_BakeCamPosition = loader.getBakeCamPosition();
		m_BakeCamLookAt = loader.getBakeCamLookAt();
		m_BakeCamUp = loader.getBakeCamUp();
		m_BakeWidth = loader.getBakeWidth();
		m_BakeHeight = loader.getBakeHeight();
		m_BakeAspect = loader.getBakeAspect();

		m_ViewProjMatrix = loader.getViewProjMatrix();
		m_InvViewProjMatrix = loader.getInvViewProjMatrix();

		m_Bounds = BBox( loader.getBoundingBox().getMin(),
						 loader.getBoundingBox().getMax() );

		m_RealPointCount = loader.getPointCount();
		m_PointCount = m_RealPointCount * ( 100 / i_Density );

		// get position
		{
			const std::vector< float >& p = loader.getPositionVec();
			assert( p.size() / 3 == m_RealPointCount );
			m_Data.getFloatParam( "P" ).setBufferType( GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW );
			m_Data.getFloatParam( "P" ).setValue( p );
		}

		// encode normal and radius into uv coordinate
		{
			m_Data.getHalfParam( "NR" ).setBufferType( GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW );
			m_Data.getHalfParam( "NR" ).setValue( loader.getParam( "NR" ) );
		}

		// now let's deal with user variable and place them into uv coordinate
		{
			m_CurrentUserDataIndex = -1;
			int user_data_size = loader.getUserDataSize();
			// for ex it would be 6 if we have 2 colors

			const std::vector< half >& o = loader.getParam( "other" );
			assert( o.size() == m_RealPointCount * user_data_size );

			int offset = 0;
			for (int iPtcVar=0; iPtcVar < loader.getUserVariableCount(); ++iPtcVar )
			{
				const std::string & ptcUserVarName = loader.getUserVariableName( iPtcVar );
				const std::string & ptcUserVarType = loader.getUserVariableType( iPtcVar );

				// restart iterator
				std::vector< half >::const_iterator oi( o.begin() );
				oi += offset;

				int user_var_size = getUserDataSizeFromType(ptcUserVarType);
				// uv coordinate are 4 elements max so a matrix for example
				// would need to be split and would use 4 uvs..

				HostVector< half > hostV3;
				hostV3.reserve( m_RealPointCount * 3 );

				for( int i = 0; i < m_RealPointCount; ++i )
				{
					for( int j = 0; j < user_var_size; ++j ) hostV3.push_back( oi[j] );

					oi += user_data_size;
				}

				m_Data.getHalfParam( ptcUserVarName.c_str() ).setBufferType( GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW );
				m_Data.getHalfParam( ptcUserVarName.c_str() ).setValue( hostV3 );

				// set it on the first one found
				if ( m_CurrentUserDataIndex == -1 ) m_CurrentUserDataIndex = iPtcVar;

				m_UserDataVecParams.push_back( std::make_pair( ptcUserVarName, ptcUserVarType ) );
				offset += user_var_size;
			}
		}

		t.stop();
		DRD_LOG_INFO( L, "pack/transfer time: " << t.elapsed_sec() << "s" );
	}
}

//-------------------------------------------------------------------------------------------------
void ParticleSystem::dumpGL( float lod ) const
{
	if ( m_RealPointCount == 0 ) return;

	const DeviceVector< float >& P = m_Data.getFloatParam( "P" );
	//const DeviceVector< Imath::V3f >& P = m_Data.getVec3fParam( "P" );

	// positions
	SAFE_GL( glEnableClientState( GL_VERTEX_ARRAY ) );
	P.bindGL();
	SAFE_GL( glVertexPointer( 3, GL_FLOAT, 0, 0 ) );

	if ( !m_PositionOnly )
	{
		const DeviceVector< half >& NR = m_Data.getHalfParam( "NR" );

		glClientActiveTextureARB(GL_TEXTURE0_ARB);

		glEnableClientState( GL_TEXTURE_COORD_ARRAY );
		NR.bindGL();
		glTexCoordPointer( 4, GL_HALF_FLOAT_NV, 0, NULL );

		if ( m_CurrentUserDataIndex != -1 )
		{
			glClientActiveTextureARB(GL_TEXTURE1_ARB);
			glEnableClientState( GL_TEXTURE_COORD_ARRAY );

			int curUserDataSize = getCurrentUserDataSize();
			//if ( curUserDataSize == 1) // float
			{
				const DeviceVector< half >& DV = m_Data.getHalfParam( getCurrentUserDataName() );
				DV.bindGL();
			}
			/*else // let's assume a vec3 type for now
			{
				const DeviceVector< Imath::V3h >& DV = m_Data.getVec3hParam( getCurrentUserDataName() );
				DV.bindGL();
			}*/

			glTexCoordPointer( curUserDataSize, GL_HALF_FLOAT_NV, 0, NULL );
		}


#if 0
		PFNGLPOINTPARAMETERFARBPROC glPointParameterfARB  = (PFNGLPOINTPARAMETERFARBPROC)glXGetProcAddressARB((GLubyte*)"glPointParameterfARB");
		PFNGLPOINTPARAMETERFVARBPROC glPointParameterfvARB = (PFNGLPOINTPARAMETERFVARBPROC)glXGetProcAddressARB((GLubyte*)"glPointParameterfvARB");

		float maxSize = 0.0f;
		glGetFloatv( GL_POINT_SIZE_MAX, &maxSize );
		glPointSize( maxSize );
		glPointParameterfARB( GL_POINT_SIZE_MAX, maxSize );
		glPointParameterfARB( GL_POINT_SIZE_MIN, 1.0f );
	//	float quadratic[] = { 1.0f, -0.01f, 10.0f };
	//	glPointParameterfvARB( GL_POINT_DISTANCE_ATTENUATION, quadratic );
		glTexEnvf( GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE );
		glEnable( GL_POINT_SPRITE );
		glEnable( GL_POINT_SMOOTH );
		//	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
#endif
		//glPointSize( 2 );
	}

	SAFE_GL( glDrawArrays( GL_POINTS, 0, P.size() ) );
#if 0
	glDisable( GL_POINT_SPRITE );
#endif

	if ( !m_PositionOnly )
	{
		const DeviceVector< half >& NR = m_Data.getHalfParam( "NR" );
		SAFE_GL( glDisableClientState( GL_TEXTURE_COORD_ARRAY ) );
		NR.unbindGL();
	}

	SAFE_GL( glDisableClientState( GL_VERTEX_ARRAY ) );
	P.unbindGL();
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
