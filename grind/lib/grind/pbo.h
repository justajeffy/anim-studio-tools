/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: $"
 * SVN_META_ID = "$Id: pbo.h 91641 2011-07-18 03:13:38Z chris.bone $"
 */

#ifndef grind_pbo_h
#define grind_pbo_h
//-------------------------------------------------------------------------------------------------
#include "pbo_types.h"
#include <boost/shared_ptr.hpp>

//-------------------------------------------------------------------------------------------------
// forward declarations
namespace bee
{
	class TextureLoader;
}

namespace grind
{

template< typename T > class DeviceVector;
template< typename T > class DeviceVectorHandle;

//-------------------------------------------------------------------------------------------------
//! a device pixel buffer object
class PBO
{
public:
	//! default constructor
	PBO();

	//! load from a file path
	void read( const std::string& path );

	void sampleMap( const DeviceVector<float>& u,
	                const DeviceVector<float>& v,
	                DeviceVector<float>& result ) const;

	//! sample at uv locations
	void sample(	const DeviceVectorHandle< float >& u,
					const DeviceVectorHandle< float >& v,
					DeviceVectorHandle< float >& result ) const;

	//! report the memory usage
	size_t memSize() const;

	//! @cond DEV

	//! default destructor
	~PBO();

	//! allow external kernels to do their own bind and access
	CudaArrayHandle& getCudaArrayHandle() { return m_CudaArrayHandle; }

private:

	//! width of loaded image
	unsigned int m_Width;

	//! height of loaded image
	unsigned int m_Height;

	//! act
	CudaArrayHandle m_CudaArrayHandle;

	//! has the file been loaded?
	bool m_FileLoaded;

	//! Texture loader used for CPU based sampling
	boost::shared_ptr< const bee::TextureLoader > m_TextureLoader;

	//! store the path
	std::string m_Path;

	//! host implementation of texture sampling
	void sampleTextureHost( const DeviceVector<float>& u,
	                        const DeviceVector<float>& v,
	                        DeviceVector<float>& result ) const;

	//! will RX be used for shading?
	bool useRx() const;

	//! @endcond
};

} // namespace Grind


#endif /* grind_pbo_h */


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
