
#include <stddef.h>
#include <sys/types.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/stat.h>
#include <errno.h>
#include <assert.h>

#include <drdDebug/log.h>
#include <drdDebug/runtimeError.h>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <boost/foreach.hpp>
#include <boost/format.hpp>

#include "io/ObjLoader.h"
#include "gl/Mesh.h"

//----------------------------------------------------------------------------
using namespace bee;
using namespace std;
using namespace drd;
DRD_MKLOGGER(L,"drd.bee.io.ObjLoader");

//----------------------------------------------------------------------------
namespace
{
	//----------------------------------------------------------------------------
	void
	skipLine( stringstream & is )
	{
		char next;
		is >> std::noskipws;
		while ( ( is >> next ) && ( '\n' != next ) )
		{
			;
		}
	}

	//----------------------------------------------------------------------------
	bool
	skipCommentLine( stringstream & is )
	{
		char next;
		while ( is >> std::skipws >> next )
		{
			is.putback( next );
			if ( '#' == next ) skipLine( is );
			else return true;
		}
		return false;
	}

	//----------------------------------------------------------------------------
	inline void
	skipLine( const char*& pBuff)
	{
		while( *(pBuff) != '\n' && *(pBuff) != 0 )
		{
			pBuff++;
		}
	}

	//----------------------------------------------------------------------------
	inline void
	skipSpace( const char*& pBuff)
	{
		while( *(pBuff) == ' ' || *(pBuff) == '\n')
		{
			++pBuff;
		}
	}
	//----------------------------------------------------------------------------
	inline bool
	skipCommentLine( const char*& pBuff)
	{
		while( true)
		{
			skipSpace(pBuff);
			if (*pBuff == 0)
				return false;

			if (*(pBuff) == '#')  skipLine(pBuff);
			else return true;
		}
		return false;
	}

	//----------------------------------------------------------------------------
	inline bool readString(const char*& pBuff,char* outString) //max string size is 255
	{
		char cnt=0;
		skipSpace(pBuff);
		while( *(pBuff) != ' ' && *(pBuff) != '\n' && *(pBuff) != 0 )
		{
			outString[cnt++]= *(pBuff++);
			assert(cnt <255);
		}
		outString[cnt]=0;
		return (cnt!=0);

	}

	//----------------------------------------------------------------------------
	inline bool readLine(const char*& pBuff,char* outString) //max string size is 255
	{
		char cnt=0;
		skipSpace(pBuff);
		while( *(pBuff) != '\n' && *(pBuff) != 0 )
		{
			outString[cnt++]= *(pBuff++);
			assert(cnt <255);
		}
		outString[cnt]=0;
		return (cnt!=0);

	}

}

//----------------------------------------------------------------------------
// static
bool
ObjLoader::is( const std::string & a_BaseFilename )
{
	string s ( a_BaseFilename );
	transform( s.begin(), s.end(), s.begin(), ::tolower );
	return s.find( ".obj" ) == s.length() - 4;
}

//----------------------------------------------------------------------------
ObjLoader::ObjLoader( const std::string & a_BaseFilename )
:	MeshLoader( a_BaseFilename, Loader::eObj )
,	m_TotalFaceCount( 0 )
,	m_Loaded( false )
,	m_CurrentFrame( 1 )
{
}

//----------------------------------------------------------------------------
bool
ObjLoader::open()
{
	if ( m_Loaded )
	{
		DRD_LOG_WARN( L, "File already loaded: "<<m_CurrentFileName );
		return true;
	}
	return load();
}

//----------------------------------------------------------------------------
#define ck(x) \
if( (x) == (-1) ){ perror("");exit(EXIT_FAILURE);}
//----------------------------------------------------------------------------
/* read a buffer from a file */
ssize_t readall(int fd, void *buf, size_t *bytes)
{
     ssize_t nread = 0, n=0;
     size_t nbyte = *bytes;

     do {
         if ((n = read(fd, &((char *)buf)[nread], nbyte - nread)) == -1) {
             if (errno == EINTR)
                 continue;
             else
                 return (-1);
         }
         if (n == 0)
             return nread;
         nread += n;
     } while (nread < nbyte);
     return nread;
}
//----------------------------------------------------------------------------
/* read control */
void readfile(const char *fname, char *&buffer, size_t *size, mode_t *mode)
{
   int fd=0;
   struct stat st;

   ck(fd=open(fname,O_RDONLY) );
   ck(fstat(fd,&st) );
   *size=st.st_size;
   *mode=st.st_mode;
   buffer=(char*)calloc(1,*size+1);
   ck(readall(fd, buffer, size) );
   ck(close(fd) );
}



//----------------------------------------------------------------------------
//#define USE_IFSTREAM
bool
ObjLoader::load()
{
	if ( m_CurrentFileName.empty() )
	{
		using namespace boost::io;
		using namespace boost;

		DRD_LOG_DEBUG( L, "Need a new filename, " << m_BaseFilename << ":" << m_CurrentFrame << ":" << m_CurrentFileName );

		format fmter( m_BaseFilename );
		fmter.exceptions( all_error_bits ^ ( too_many_args_bit | too_few_args_bit )  );
		fmter % m_CurrentFrame;
		m_CurrentFileName = fmter.str();
	}

	if ( m_Loaded )
	{
		DRD_LOG_WARN( L, "File already loaded: "+m_CurrentFileName );
		return true;
	}

	DRD_LOG_INFO( L, "Loading file: "+m_CurrentFileName );

	//hold the file in tmp buffer
	//
#ifndef USE_IFSTREAM
	char *buffer=NULL;
	size_t size;
	mode_t mode;

	readfile(m_CurrentFileName.c_str(),buffer,&size,&mode);
	assert(buffer !=NULL);

	// do parsing
	const char* const_pBuff = buffer;
	while( skipCommentLine(const_pBuff) )
	{
		if ( false == processLine( const_pBuff ) )
			break;
	}

	free(buffer);
	m_Loaded = true;

	return true;
	//
#else
	// Check that file exists
	ifstream ifs( m_CurrentFileName.c_str(), ios::in|ios::ate );
	if( !ifs )
	{
		DRD_LOG_ERROR( L, "File does not exist or can't open: "<<m_CurrentFileName );
		return false;
	}

	// Check file size
	const int fileSize = ifs.tellg();
	ifs.seekg( 0, ios::beg );
	if( fileSize <= 0 )
	{
		throw drd::RuntimeError( "File is empty: "+m_CurrentFileName );
		return false;
	}


	stringstream ss;
	ss << ifs.rdbuf(); // Put the whole file into ss
	// use ss as every other stream...

	while( skipCommentLine(ss) )
	{
		if ( false == processLine( ss ) )
			break;
	}

	m_Loaded = true;

	return true;
#endif

}

//----------------------------------------------------------------------------
bool
ObjLoader::write()
{
	throw drd::RuntimeError( "Writing is not yet supported" );
	return false;
}

//----------------------------------------------------------------------------
bool
ObjLoader::close()
{
	// this file is always closed once it has been either read or written.
	return true;
}

//----------------------------------------------------------------------------
bool
ObjLoader::processLine( stringstream & is )
{
	String ele_id;
	float x=0.0f, y=0.0f, z=0.0f;

	if ( !( is >> ele_id ) )
		return false;

	if ( "mtllib" == ele_id )
	{
		std::string strFileName;
		is >> strFileName;
		DRD_LOG_INFO( L, "ignoring material file " << strFileName );
	}
	else if ("usemtl" == ele_id)
	{
		std::string strMtlName;
		is >> strMtlName;
		DRD_LOG_INFO( L, "ignoring material file " << strMtlName );
	}
	else if ("v" == ele_id) //	vertex data
	{
		is >> x >> y >> z;
		m_VertexVector.push_back( Imath::V3f( x, y, z ) );
	}
	else if ("vt" == ele_id) // texture data
	{
		is >> x >> y >> z;
		is.clear(); // is z (i.e. w) is not available, have to clear error flag.
		//cout << "UV " << x << " " << y << endl;
		m_TexCoordVector.push_back( Imath::V2f( x, y ) );
	}
	else if ("vn" == ele_id) // normal data
	{
		is >> x >> y >> z;
		if (!is.good())
		{ // in case it is -1#IND00
			x = y = z = 0.0;
			is.clear();
			skipLine(is);
		}
		m_NormalVector.push_back( Imath::V3f( x, y, z ) );
	}
	else if ("f" == ele_id) //	face data
	{
		bool containsNormals = m_NormalVector.size() > 0;
		bool containsTexCoords = m_TexCoordVector.size() > 0;

		//	face treatment
		//  Note: obviously this technique only works for convex polygons with ten verts or less.
		// TODO: fix this stuff at some point !
		int vi[10]; // vertex indices.
		int ni[10] = { 0, 0, 0, 0, }; // normal indices.
		int ti[10] = { 0, 0, 0, 0, }; // tex indices.
		int i = 0;

		for (char c; i < 10; )
		{
			if ( !containsTexCoords && !containsNormals )
				is >> vi[i];
			else if ( !containsTexCoords )
				is >> vi[i] >> c >> c >> ni[i];
			else if ( !containsNormals )
				is >> vi[i] >> c >> ti[i];
			else
				is >> vi[i] >> c >> ti[i] >> c >> ni[i];

			if (!is.good())
				break;
			++i;
		}


		//	Create the polygon face
		m_PolygonVector.push_back( MeshPolygon( i ) );
		m_TotalFaceCount += i-2;
		if( i <= 2 )
		{
			throw drd::RuntimeError( "File contains bad faces!" );
		}

		MeshPolygon& polygon( m_PolygonVector.back() );
		for (int k = 0; k < i; ++k)
		{
			polygon.m_FaceVector.push_back( MeshFace( vi[k] - 1, ni[k] - 1, ti[k] - 1 ) );
		}

		is.clear();
	}
	else
	{
		skipLine(is);
	}
	return true;
}

//----------------------------------------------------------------------------
bool
ObjLoader::processLine( const char*& pBuffer)
{

	bool stat;
	char stringBuff[255];
	float x=0.0f, y=0.0f, z=0.0f;

	stat = readString(pBuffer,stringBuff);
	if ( !(stat ) )
		return false;

	if ( !strcmp("mtllib", stringBuff ) )
	{
		stat = readString(pBuffer,stringBuff);
		assert (stat);
		DRD_LOG_INFO( L, "ignoring material file " << stringBuff );
	}
	else if (  !strcmp("usemtl", stringBuff ) )
	{
		stat = readString(pBuffer,stringBuff);
		assert (stat);
		DRD_LOG_INFO( L, "ignoring material file " << stringBuff );
	}
	else if ( !strcmp("v", stringBuff ) ) //	vertex data
	{
		stat = readString(pBuffer,stringBuff);
		if ( !(stat ) )
			return false;
		 x = atof(stringBuff);
		 stat = readString(pBuffer,stringBuff);
		 if ( !(stat ) )
			return false;
		 y = atof(stringBuff);
		 stat = readString(pBuffer,stringBuff);
		 if ( !(stat ) )
			return false;
		 z = atof(stringBuff);

		m_VertexVector.push_back( Imath::V3f( x, y, z ) );
	}
	else if ( !strcmp("vt", stringBuff ) ) // texture data
	{

		stat = readLine(pBuffer,stringBuff);
		if ( !(stat ) ) return false;
		std::string bs(stringBuff);
		std::vector<std::string> aKey;
		boost::split(aKey, bs, boost::is_any_of(" "));
		assert(aKey.size()==2 || aKey.size()==3);

		x = atof(aKey[0].c_str());
		y = atof(aKey[1].c_str());

		m_TexCoordVector.push_back( Imath::V2f( x, y ) );
	}
	else if (  !strcmp("vn", stringBuff ) ) // normal data
	{
		stat = readString(pBuffer,stringBuff);
		if ( !(stat ) ) return false;
		x = atof(stringBuff);
		stat = readString(pBuffer,stringBuff);
		if ( !(stat ) ) return false;
		y = atof(stringBuff);
		stat = readString(pBuffer,stringBuff);
		if ( !(stat ) ) return false;
		z = atof(stringBuff);

		m_NormalVector.push_back( Imath::V3f( x, y, z ) );
	}
	else if ( !strcmp("f", stringBuff ) ) //	face data
	{
		bool containsNormals = m_NormalVector.size() > 0;
		bool containsTexCoords = m_TexCoordVector.size() > 0;

		//	face treatment
		//  Note: obviously this technique only works for convex polygons with ten verts or less.
		// TODO: fix this stuff at some point !
		int vi[10]; // vertex indices.
		int ni[10] = { 0, 0, 0, 0, }; // normal indices.
		int ti[10] = { 0, 0, 0, 0, }; // tex indices.
		int i = 0;

		for (char c; i < 10; )
		{
			if ( !containsTexCoords && !containsNormals )
			{
				stat = readString(pBuffer,stringBuff);
				if ( !(stat ) ) return false;
				vi[i] = atoi(stringBuff);
			}
			else if ( !containsTexCoords )
			{
				stat = readString(pBuffer,stringBuff);
				if ( !(stat ) ) return false;

				std::string bs(stringBuff);
				std::vector<std::string> aKey;
				boost::split(aKey, bs, boost::is_any_of("//"));
				assert(aKey.size()==2);

				vi[i] = atoi(aKey[0].c_str());
				ni[i] = atoi(aKey[1].c_str());
			}
			else if ( !containsNormals )
			{
				stat = readString(pBuffer,stringBuff);
				if ( !(stat ) ) return false;


				std::string bs(stringBuff);
				std::vector<std::string> aKey;
				boost::split(aKey, bs, boost::is_any_of("/"));
				assert(aKey.size()==2);

				vi[i] = atoi(aKey[0].c_str());
				ti[i] = atoi(aKey[1].c_str());


			}
			else
			{
				stat = readString(pBuffer,stringBuff);
				if ( !(stat ) ) return false;

				std::string bs(stringBuff);

				std::vector<std::string> aKey;
				boost::split(aKey, bs, boost::is_any_of("/"));
				assert( aKey.size()==3 );

				vi[i] = atoi(aKey[0].c_str());
				ti[i] = atoi(aKey[1].c_str());
				ni[i] = atoi(aKey[2].c_str());

			}
			while( *(pBuffer) == ' ' ) ++pBuffer;
			if (*pBuffer =='\n' || (*pBuffer)==0)
				break;
			++i;
		}
		++i;
		//	Create the polygon face
		m_PolygonVector.push_back( MeshPolygon( i ) );
		m_TotalFaceCount += i-2;
		if( i <= 2 )
		{
			throw drd::RuntimeError( "File contains bad faces!" );
		}

		MeshPolygon& polygon( m_PolygonVector.back() );
		for (int k = 0; k < i; ++k)
		{
			polygon.m_FaceVector.push_back( MeshFace( vi[k] - 1, ni[k] - 1, ti[k] - 1 ) );
		}

	}
	else
	{
		skipLine(pBuffer);
	}
	return true;
}
//----------------------------------------------------------------------------
boost::shared_ptr< Mesh >
ObjLoader::createMesh()
{
	boost::shared_ptr< Mesh > mesh( new Mesh( m_TotalFaceCount * 3 ) );
	bool containsNormal = !m_NormalVector.empty();
	bool containsTexCoord = !m_TexCoordVector.empty();

	auto_ptr< Imath::V3f > pVertexData ( new Imath::V3f[ m_TotalFaceCount * 3 ] );
	auto_ptr< Imath::V3f > pNormalData;
	auto_ptr< Imath::V2f > pTexCoordData;

	if ( containsNormal )
		pNormalData.reset( new Imath::V3f[ m_TotalFaceCount * 3 ] );
	if ( containsTexCoord )
		pTexCoordData.reset( new Imath::V2f[ m_TotalFaceCount * 3 ] );

	Imath::V3f * pCurVD = pVertexData.get();
	Imath::V3f * pCurND = pNormalData.get();
	Imath::V2f * pCurTCD = pTexCoordData.get();

	int curVert = 0;
	BOOST_FOREACH( const MeshPolygon & poly, m_PolygonVector )
	{
		for (int i = 2; i < poly.m_FaceVector.size(); ++i)
		{
			// Add our 3 vertices
			int vertexIdx0 = poly.m_FaceVector[0].vertexIdx;
			int vertexIdx1 = poly.m_FaceVector[i-1].vertexIdx;
			int vertexIdx2 = poly.m_FaceVector[i].vertexIdx;

			if( vertexIdx0 < 0
			 || vertexIdx1 < 0
			 || vertexIdx2 < 0
			 || vertexIdx0 >= m_VertexVector.size()
			 || vertexIdx1 >= m_VertexVector.size()
			 || vertexIdx2 >= m_VertexVector.size() )
				throw drd::RuntimeError( "invalid vertex index !" );


			*pCurVD++ = m_VertexVector[ vertexIdx0 ];
			*pCurVD++ = m_VertexVector[ vertexIdx1 ];
			*pCurVD++ = m_VertexVector[ vertexIdx2 ];

			// Add our 3 normals
			if ( containsNormal )
			{
				int normalIdx0 = poly.m_FaceVector[0].normalIdx;
				int normalIdx1 = poly.m_FaceVector[i-1].normalIdx;
				int normalIdx2 = poly.m_FaceVector[i].normalIdx;

				if( normalIdx0 < 0
				 || normalIdx1 < 0
				 || normalIdx2 < 0
				 || normalIdx0 >= m_NormalVector.size()
				 || normalIdx1 >= m_NormalVector.size()
				 || normalIdx2 >= m_NormalVector.size() )
					throw drd::RuntimeError( "invalid normal index !" );

				*pCurND++ = m_NormalVector[ normalIdx0 ];
				*pCurND++ = m_NormalVector[ normalIdx1 ];
				*pCurND++ = m_NormalVector[ normalIdx2 ];
			}

			// And now our 3 texCoords
			if ( containsTexCoord )
			{
				int texCoordIdx0 = poly.m_FaceVector[0].texCoordIdx;
				int texCoordIdx1 = poly.m_FaceVector[i-1].texCoordIdx;
				int texCoordIdx2 = poly.m_FaceVector[i].texCoordIdx;

				if( texCoordIdx0 < 0
				 || texCoordIdx1 < 0
				 || texCoordIdx2 < 0
				 || texCoordIdx0 >= m_TexCoordVector.size()
				 || texCoordIdx1 >= m_TexCoordVector.size()
				 || texCoordIdx2 >= m_TexCoordVector.size() )
					throw drd::RuntimeError( "invalid tex coord index !" );

				*pCurTCD++ = m_TexCoordVector[ texCoordIdx0 ];
				*pCurTCD++ = m_TexCoordVector[ texCoordIdx1 ];
				*pCurTCD++ = m_TexCoordVector[ texCoordIdx2 ];
			}
		}
	}

	// Memory overflow check
	if( ( pCurVD != pVertexData.get() + m_TotalFaceCount * 3 )
	 || ( containsNormal && pCurND != pNormalData.get() + m_TotalFaceCount * 3 )
	 || ( containsTexCoord && pCurTCD != pTexCoordData.get() + m_TotalFaceCount * 3 ) )
		throw drd::RuntimeError( "Memory overflow" );

	// Fill the created mesh
	mesh->createVertexBuffer( 3, sizeof( float ), pVertexData.get() );
	if ( containsNormal ) mesh->createNormalBuffer( 3, sizeof( float ), pNormalData.get() );
	if ( containsTexCoord ) mesh->createTexCoordBuffer( 2, sizeof( float ), pTexCoordData.get() );

	return mesh;
}


//----------------------------------------------------------------------------
void
ObjLoader::reportStats()
{
	DRD_LOG_INFO(L, "ObjLoader Report :" );
	DRD_LOG_INFO(L, "- FileName : " << m_CurrentFileName );
	DRD_LOG_INFO(L, "- vertex count : " << m_VertexVector.size() );
	DRD_LOG_INFO(L, "- normal count : " << m_NormalVector.size() );
	DRD_LOG_INFO(L, "- texCoord count : " << m_TexCoordVector.size() );
	DRD_LOG_INFO(L, "- polygon count : " << m_PolygonVector.size() );
	DRD_LOG_INFO(L, "- face count : " << m_TotalFaceCount );
}

//----------------------------------------------------------------------------
bool
ObjLoader::setFrame( float a_Frame )
{
	int frame = static_cast< int >( a_Frame );

	DRD_LOG_DEBUG(L, "setFrame:" << a_Frame << ":" << frame);

	if ( m_CurrentFrame == frame )
		return m_Loaded;

	m_CurrentFrame = frame;

	m_VertexVector.clear();
	m_NormalVector.clear();
	m_TexCoordVector.clear();
	m_PolygonVector.clear();

	m_CurrentFileName = "";
	m_Loaded = false;

	return load();
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
