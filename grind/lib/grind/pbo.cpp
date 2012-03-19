
//-------------------------------------------------------------------------------------------------
#include <drdDebug/log.h>
DRD_MKLOGGER(L,"drd.grind.PBO");

#include "pbo.h"
#include "log.h"
#include "host_vector.h"
#include "device_vector.h"
#include "timer.h"
#include <bee/gl/PBO.h>
#include <bee/io/textureLoader.h>

#include <cutil_inline.h>

#include <GL/gl.h>

#include <cutil_inline.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

#include <rx.h>

#include <boost/thread/mutex.hpp>

#define ALLOW_RX_SHADING 0

//-------------------------------------------------------------------------------------------------
// device code
extern "C" void initTextureDevice(int imageWidth, int imageHeight, const void * h_data, CudaArrayHandle& handle );
extern "C" void freeTextureDevice( CudaArrayHandle& handle );
extern "C" void sampleTextureDevice( const CudaArrayHandle& handle, unsigned int n, float* u, float* v, float* result );

//-------------------------------------------------------------------------------------------------
using namespace grind;

static boost::shared_ptr<const bee::TextureLoader> getTexture(const std::string &image_path)
{
	static boost::mutex mx;
	boost::mutex::scoped_lock sl(mx);
	static std::map<std::string, boost::weak_ptr<bee::TextureLoader> > texture_cache;

	boost::shared_ptr<bee::TextureLoader> textureLoader;
	std::map<std::string, boost::weak_ptr<bee::TextureLoader> >::iterator i = texture_cache.find(image_path);
	if(i != texture_cache.end())
	{
		textureLoader = (*i).second.lock();
	}

	if(!textureLoader) // have to load, or reload, the texture
	{
//		std::cerr << "PBO IMAGE CACHE MISS: " << image_path << std::endl;
		textureLoader.reset( new bee::TextureLoader(image_path) );
		texture_cache[image_path] = textureLoader;
	}
	else
	{
//		std::cerr << "PBO IMAGE CACHE HIT: " << image_path << std::endl;
	}
	return textureLoader;
}

//-------------------------------------------------------------------------------------------------
void PBO::read( const std::string& image_path )
{
	m_Path = image_path;

#ifdef __DEVICE_EMULATION__
	if( !useRx() ){
		m_TextureLoader = getTexture(image_path);
		m_Width = m_TextureLoader->getWidth();
		m_Height = m_TextureLoader->getHeight();
		DRD_LOG_DEBUG( L, "read " << memSize() / 1000000 << "Mb image from " << m_Path );
	}

	// don't need to do any prep for emulation mode
#else
	bee::TextureLoader loader( image_path );
	loader.reportStats();

	m_Width = loader.getWidth();
	m_Height = loader.getHeight();
	DRD_LOG_ASSERT( L, m_Width>0&&m_Height>0, "Bad texture file: " << image_path );
	const void * host_data = loader.getBuffer();
	DRD_LOG_INFO( L, "Image [" << image_path << "] : " << m_Width << "x" << m_Height );

	initTextureDevice( m_Width, m_Height, host_data, m_CudaArrayHandle );
#endif

    m_FileLoaded = true;
}

//-------------------------------------------------------------------------------------------------
//! sample a texture via a renderman texture sampler
void sampleRx( const std::string& filename, int count, const float* u, const float* v, float* dst )
{
	if( count == 0 ) return;

	// query the image resolution
	std::vector< float > res( 2, 0.0f );
	RxInfoType_t o_resulttype;
	RtInt o_resultcount;
	RtInt error = RxTextureInfo( filename.c_str(), "resolution", &( res[ 0 ] ), sizeof(float) * 2, &o_resulttype, &o_resultcount );
	assert( !error );
	//std::cout << "res: " << res[0] << "," << res[1] << std::endl;

	// derivatives for sampling
	float dx = 0.25f / res[ 0 ];
	float dy = 0.25f / res[ 1 ];

	char* filter_type = "triangle";

	// now sample from the texture
	for ( int i = 0 ; i < count ; ++i )
	{
		float uu = u[ i ];
		float vv = 1.0f - v[ i ];
		RtInt e = RxTexture( filename.c_str(), 0, 1,
				uu-dx, vv-dy,
				uu-dx, vv+dy,
				uu+dx, vv-dy,
				uu+dx, vv+dy,
				&( dst[ i ] ),
				"filter", &filter_type,
				RI_NULL );
		assert( !e );
	}
}

//-------------------------------------------------------------------------------------------------
void PBO::sampleMap(	const DeviceVector< float >& u,
						const DeviceVector< float >& v,
						DeviceVector< float >& result ) const
{
	DRD_LOG_ASSERT( L, m_FileLoaded, "PBO: tried to sample when texture not loaded" );
	DRD_LOG_ASSERT( L, u.isOk(), "deviceVector 'U' is not ok" );
	DRD_LOG_ASSERT( L, v.isOk(), "deviceVector 'V' is not ok" );
	DRD_LOG_ASSERT( L, result.isOk(), "deviceVector 'result' is not ok" );

	// verify array sizes match
	DRD_LOG_ASSERT( L, u.size() == v.size() && u.size() == result.size(), "array sizes must match for PBO::sampleTexture()" );

#ifdef __DEVICE_EMULATION__
	if( useRx() ){
		DRD_LOG_DEBUG( L, "sampling using rx: " << m_Path );
		sampleRx( m_Path, result.size(), u.getDevicePtr(), v.getDevicePtr(), result.getDevicePtr() );
		return;
	} else {
		DRD_LOG_DEBUG( L, "sampling using cpu: " << m_Path );
		sampleTextureHost( u, v, result );
	}
#else
	DRD_LOG_DEBUG( L, "sampling using cuda: " << m_Path );
	// now sample on the device
	sampleTextureDevice( m_CudaArrayHandle, result.size(), u.getDevicePtr(), v.getDevicePtr(), result.getDevicePtr() );
#endif

}


//-------------------------------------------------------------------------------------------------
void PBO::sample( const DeviceVectorHandle<float>& uh
                , const DeviceVectorHandle<float>& vh
                , DeviceVectorHandle<float>& resulth ) const
{
	const DeviceVector<float>& u = uh.get();
	const DeviceVector<float>& v = vh.get();
	DeviceVector<float>& result = resulth.get();

	sampleMap( u, v, result );
}


#if 0
//-------------------------------------------------------------------------------------------------
void PBO::test()
{
	HostVector< float > hResult, hU, hV;
	DeviceVector< float > result, u, v;

	size_t n_samples = 100;

	hResult.resize( n_samples, 0.0f );
	result.setValue(hResult);

	hU.resize( n_samples,0.5f);
	hV.resize( n_samples,0.5f);

	for( size_t i =0; i < n_samples; ++i ){
		hU[i] = nRand();
		hV[i] = nRand();
	}


	u.setValue( hU );
	v.setValue( hV );

	DRD_LOG_DEBUG( L, "before:" );
	result.dump();

	sampleTextureDevice( m_Width, m_Height, result.size(), u.getDevicePtr(), v.getDevicePtr(), result.getDevicePtr() );

	DRD_LOG_DEBUG( L, "after:" );
	result.dump();

	DRD_LOG_DEBUG( L, "finished constructing" );
}
#endif

//-------------------------------------------------------------------------------------------------
PBO::PBO()
: m_FileLoaded( false )
{
}

//-------------------------------------------------------------------------------------------------
PBO::~PBO()
{
#ifdef __DEVICE_EMULATION__
#else
	freeTextureDevice( m_CudaArrayHandle );
#endif
}


//-------------------------------------------------------------------------------------------------
//! cpu emulation of cuda texture sampling as per appendix E of nvidia programming guide
struct CpuTextureSampler
{
	struct Pixel{ unsigned char r,g,b,a; };

	inline Pixel& getTexel( int x, int y )
	{
		// clamp to range
		if( x < 0 ) x=0;
		if( x >= m_Width ) x = m_Width-1;
		if( y < 0 ) y=0;
		if( y >= m_Height ) y = m_Height-1;

		return m_Data[ x + y * m_Width ];
	}

	CpuTextureSampler( int i_Width, int i_Height, void* data )
	: m_Width( i_Width )
	, m_Height( i_Height )
	, m_Data( (Pixel*) data )
	{}


	//! bilinear interpolation using floating point pixel values (not normalized)
	float bilin( float x, float y )
	{
		float xb = x-0.5f;
		int i = int(floor(xb));
		float alpha = frac(xb);

		float yb = y-0.5f;
		int j = int(floor(yb));
		float beta = frac(yb);

		float t0 = float( getTexel(i  ,j  ).r ) / 255.0f;
		float t1 = float( getTexel(i+1,j  ).r ) / 255.0f;
		float t2 = float( getTexel(i  ,j+1).r ) / 255.0f;
		float t3 = float( getTexel(i+1,j+1).r ) / 255.0f;

		return (1.0f - alpha)*(1.0f - beta)*t0
			 + (       alpha)*(1.0f - beta)*t1
		     + (1.0f - alpha)*(       beta)*t2
		     + (       alpha)*(       beta)*t3;
	}

	inline float frac( float v ) { return v - floor( v ); }

	void sample( int n, const float* u, const float* v, float* result )
	{
#pragma omp parallel for
		for( int i = 0; i < n; ++i ){
			float x = u[i] * float(m_Width);
			float y = v[i] * float(m_Height);
			result[i] = bilin( x, y );
		}
	}

private:
	int m_Width;
	int m_Height;
	Pixel* m_Data;
};


//-------------------------------------------------------------------------------------------------
void PBO::sampleTextureHost( const DeviceVector<float>& u,
 	                        const DeviceVector<float>& v,
 	                        DeviceVector<float>& result ) const
{
	assert( result.getState() == DeviceVector<float>::DV_HOST );

	// make sure the texture loader has been set up
	assert( m_TextureLoader != NULL );

	CpuTextureSampler sampler( m_TextureLoader->getWidth(), m_TextureLoader->getHeight(), m_TextureLoader->getBuffer() );
	sampler.sample( result.size(), u.getDevicePtr(), v.getDevicePtr(), result.getDevicePtr() );
}


//-------------------------------------------------------------------------------------------------
bool PBO::useRx() const
{
	// we need an RX renderman context
	return ALLOW_RX_SHADING && grind::ContextInfo::instance().hasRX();
}


//-------------------------------------------------------------------------------------------------
size_t PBO::memSize() const
{
	// renderman handling the memory
	if( useRx() ) return 0;

	// grind handling the memory, calc memory usage in bytes
	return m_Width * m_Height * sizeof(unsigned char) * 4;
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
