
//----------------------------------------------------------------------------
#include <drdDebug/log.h>
#include <FreeImage.h>
#include "TextureTools.h"
#include <GL/glew.h>
#include <GL/glut.h>
#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>

//----------------------------------------------------------------------------
using namespace bee;
using namespace drd;
using namespace std;
DRD_MKLOGGER( L, "drd.bee.io.TextureTools" );

//----------------------------------------------------------------------------
struct FreeImageWriter
{
	FreeImageWriter( FIBITMAP* _bitmap, const string& _filename, bool _QualitySuperb )
	: bitmap(_bitmap)
	, a_FileName( _filename )
	, a_QualitySuperb( _QualitySuperb )
	{}

	void operator()()
	{
		FreeImage_Save( FIF_JPEG, bitmap, a_FileName.c_str(), (a_QualitySuperb)?(JPEG_QUALITYSUPERB):(JPEG_DEFAULT) );
		FreeImage_Unload( bitmap );
		DRD_LOG_INFO( L, "done writing " << a_FileName );
	}

	FIBITMAP * bitmap;
	const string a_FileName;
	bool a_QualitySuperb;
};

//----------------------------------------------------------------------------
bool
bee::SaveGLScreenShot( const string & a_FileName, unsigned short a_Width, unsigned short a_Height, bool a_QualitySuperb )
{
	FIBITMAP * bitmap =  FreeImage_Allocate( a_Width, a_Height, 24 );
	void * fibuf = FreeImage_GetBits( bitmap );
	glReadPixels( 0, 0, a_Width, a_Height, GL_BGR, GL_UNSIGNED_BYTE, fibuf );

#if 1
	// compression and file io on a new thread, deallocation when done
	DRD_LOG_INFO( L,"Writing out '" << a_FileName << "' " << a_Width << "x" << a_Height << " (async)" );
	FreeImageWriter writer( bitmap, a_FileName, a_QualitySuperb );
	boost::thread thread( writer );
#else
	DRD_LOG_INFO( L,"Writing out '" << a_FileName << "' " << a_Width << "x" << a_Height );
	FreeImageWriter writer( bitmap, a_FileName, a_QualitySuperb );
	writer();
#endif
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
