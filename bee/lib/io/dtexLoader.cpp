/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: http://svn.drd.int/drd/apps/bee/trunk/lib/io/objLoader.cpp $"
 * SVN_META_ID = "$Id: dtexLoader.cpp 103652 2011-09-19 08:34:41Z stephane.bertout $"
 */

#include "io/dtexLoader.h"

#include <drdDebug/log.h>
DRD_MKLOGGER(L,"drd.bee.io.DtexLoader");

#include <boost/filesystem.hpp>
#include <iostream>

#include "gl/Mesh.h"

using namespace bee;
using namespace std;

//-------------------------------------------------------------------------------------------------
DtexLoader::DtexLoader()
: m_PointCount(0)
, m_File( NULL )
, m_Image( NULL )
, m_Width( 0 ),
m_Height( 0 ),
m_ChannelCount( 0 ),
m_TileWidth( 0 ),
m_TileHeight( 0 )
{
}

//-------------------------------------------------------------------------------------------------
DtexLoader::~DtexLoader()
{
	close();
}

//-------------------------------------------------------------------------------------------------
void DtexLoader::open( const std::string& i_FilePath )
{
	DRD_LOG_INFO( L, "Opening file: "+i_FilePath );
	m_FilePath = i_FilePath;

	if ( !boost::filesystem::exists( i_FilePath ) )
		throw std::runtime_error( std::string( "DtexLoader: file doesn't exist " ) + i_FilePath );

    if ( DtexOpenFile( i_FilePath.c_str(), "rb", NULL, &m_File ) != DTEX_NOERR )
		throw std::runtime_error( "DtexLoader failed to load DSM" );

    m_ImageCount = DtexCountImages( m_File );
    if ( m_ImageCount == 0 )
		throw std::runtime_error( "DtexLoader failed: error no image" );

    if ( DtexGetImageByIndex(m_File, 0, &m_Image) != DTEX_NOERR )
		throw std::runtime_error( "DtexLoader failed: error in DtexGetImageByIndex" );

    m_Width = DtexWidth( m_Image );
    m_Height = DtexHeight( m_Image );
    m_ChannelCount = DtexNumChan( m_Image );
    m_TileWidth = DtexTileWidth( m_Image );
    m_TileHeight = DtexTileHeight( m_Image );

	// first loop to get the exact point count
	m_PointCount = 0;

	DtexPixel * pixel = DtexMakePixel( m_ChannelCount );
	unsigned tilePosX, tilePosY;

	for ( tilePosY = 0; tilePosY < m_Height; tilePosY += m_TileHeight )
	{
		for ( tilePosX = 0; tilePosX < m_Width; tilePosX += m_TileWidth )
		{
			unsigned x, y;
			for ( y = 0; y < m_TileHeight; ++y )
			{
				for (x = 0; x < m_TileWidth; ++x )
				{
					int xx = tilePosX + x;
					int yy = tilePosY + y;
					if ( xx >= m_Width || yy >= m_Height ) continue;

					DtexGetPixel( m_Image, xx, yy, pixel );
					m_PointCount += DtexPixelGetNumPoints( pixel );
				}
			}
		}
	}

	DRD_LOG_INFO( L, "\t width= " << m_Width << ", height= " << m_Height );
	DRD_LOG_INFO( L, "\t tileWidth= " << m_TileWidth << ", tileHeight= " << m_TileHeight );
	DRD_LOG_INFO( L, "\t imageCount= " << m_ImageCount << ", pointCount= " << m_PointCount );
}

//-------------------------------------------------------------------------------------------------
void DtexLoader::read()
{
	if ( m_PointCount == 0 ) return;

	m_Data.reserve( m_PointCount ); // x y z

	DtexPixel * pixel = DtexMakePixel( m_ChannelCount );
	unsigned int tilePosX, tilePosY;

	for ( tilePosY = 0; tilePosY < m_Height; tilePosY += m_TileHeight )
	{
		for ( tilePosX = 0; tilePosX < m_Width; tilePosX += m_TileWidth )
		{
			unsigned int x, y;
			for ( y = 0; y < m_TileHeight; ++y )
			{
				for (x = 0; x < m_TileWidth; ++x )
				{
					int xx = tilePosX + x;
					int yy = tilePosY + y;
					if ( xx >= m_Width || yy >= m_Height ) continue;

					DtexGetPixel( m_Image, xx, yy, pixel );

					int np = DtexPixelGetNumPoints( pixel );

					for ( unsigned int n = 0; n < np; ++n )
					{
						Vec3 p;
						float d[3];

						p.x = ((float)(xx));
						p.y = ((float)(yy));

						DtexPixelGetPoint(pixel, n, &p.z, d);

						m_Data.push_back( p );
						m_BoundingBox.update( p );
					}
				}
			}
		}
	}

	DRD_LOG_INFO( L, "read ptex data" );
}


//-------------------------------------------------------------------------------------------------
void DtexLoader::close()
{
	if( m_File == NULL ) return;
	DtexClose( m_File );
}

//-------------------------------------------------------------------------------------------------
boost::shared_ptr< Mesh > DtexLoader::createMesh()
{
	boost::shared_ptr< Mesh > mesh( new Mesh( m_PointCount ) );

	mesh->createVertexBuffer( 1, sizeof( Vec3), &(m_Data[0]) );

	return mesh;
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
