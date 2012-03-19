/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: $"
 * SVN_META_ID = "$Id: vector_set.h 42544 2010-08-17 04:31:03Z allan.johns $"
 */

#ifndef grind_device_vector_set_h
#define grind_device_vector_set_h

#include "cuda_types.h"
#include <string>
#include <map>
#include <vector>


namespace grind {

// pre-declare
template< typename T > class DeviceVector;
template< typename T > class HostVector;

//! a container for host/device vectors
template<	typename INT_VEC,
			typename FLOAT_VEC,
			typename HALF_VEC,
			typename VEC2F_VEC,
			typename VEC3F_VEC,
			typename VEC3H_VEC,
			typename VEC4F_VEC,
			typename VEC4H_VEC >
class VectorSet
{
public:

	bool hasIntParam( const std::string& name ) const;
	void getIntParamNames( std::vector< std::string >& o_Result ) const;
	INT_VEC& getIntParam( const std::string& i_Name );
	const INT_VEC& getIntParam( const std::string& i_Name ) const;

	bool hasFloatParam( const std::string& name ) const;
	void getFloatParamNames( std::vector< std::string >& o_Result ) const;
	FLOAT_VEC& getFloatParam( const std::string& i_Name );
	const FLOAT_VEC& getFloatParam( const std::string& i_Name ) const;

	bool hasHalfParam( const std::string& name ) const;
	void getHalfParamNames( std::vector< std::string >& o_Result ) const;
	HALF_VEC& getHalfParam( const std::string& i_Name );
	const HALF_VEC& getHalfParam( const std::string& i_Name ) const;

	bool hasVec2fParam( const std::string& name ) const;
	void getVec2fParamNames( std::vector< std::string >& o_Result ) const;
	VEC2F_VEC& getVec2fParam( const std::string& i_Name );
	const VEC2F_VEC& getVec2fParam( const std::string& i_Name ) const;

	bool hasVec3fParam( const std::string& name ) const;
	void getVec3fParamNames( std::vector< std::string >& o_Result ) const;
	VEC3F_VEC& getVec3fParam( const std::string& i_Name );
	const VEC3F_VEC& getVec3fParam( const std::string& i_Name ) const;

	bool hasVec3hParam( const std::string& name ) const;
	void getVec3hParamNames( std::vector< std::string >& o_Result ) const;
	VEC3H_VEC& getVec3hParam( const std::string& i_Name );
	const VEC3H_VEC& getVec3hParam( const std::string& i_Name ) const;

	bool hasVec4fParam( const std::string& name ) const;
	void getVec4fParamNames( std::vector< std::string >& o_Result ) const;
	VEC4F_VEC& getVec4fParam( const std::string& i_Name );
	const VEC4F_VEC& getVec4fParam( const std::string& i_Name ) const;

	bool hasVec4hParam( const std::string& name ) const;
	void getVec4hParamNames( std::vector< std::string >& o_Result ) const;
	VEC4H_VEC& getVec4hParam( const std::string& i_Name );
	const VEC4H_VEC& getVec4hParam( const std::string& i_Name ) const;

protected:

	typedef typename std::map< std::string, INT_VEC >						IntMapType;
	typedef typename std::map< std::string, INT_VEC >::iterator				IntMapIter;
	typedef typename std::map< std::string, INT_VEC >::const_iterator		IntMapCIter;

	typedef typename std::map< std::string, FLOAT_VEC >						FloatMapType;
	typedef typename std::map< std::string, FLOAT_VEC >::iterator			FloatMapIter;
	typedef typename std::map< std::string, FLOAT_VEC >::const_iterator		FloatMapCIter;

	typedef typename std::map< std::string, HALF_VEC >						HalfMapType;
	typedef typename std::map< std::string, HALF_VEC >::iterator			HalfMapIter;
	typedef typename std::map< std::string, HALF_VEC >::const_iterator		HalfMapCIter;

	typedef typename std::map< std::string, VEC2F_VEC >						Vec2fMapType;
	typedef typename std::map< std::string, VEC2F_VEC >::iterator			Vec2fMapIter;
	typedef typename std::map< std::string, VEC2F_VEC >::const_iterator		Vec2fMapCIter;

	typedef typename std::map< std::string, VEC3F_VEC >						Vec3fMapType;
	typedef typename std::map< std::string, VEC3F_VEC >::iterator			Vec3fMapIter;
	typedef typename std::map< std::string, VEC3F_VEC >::const_iterator		Vec3fMapCIter;

	typedef typename std::map< std::string, VEC3H_VEC >						Vec3hMapType;
	typedef typename std::map< std::string, VEC3H_VEC >::iterator			Vec3hMapIter;
	typedef typename std::map< std::string, VEC3H_VEC >::const_iterator		Vec3hMapCIter;

	typedef typename std::map< std::string, VEC4F_VEC >						Vec4fMapType;
	typedef typename std::map< std::string, VEC4F_VEC >::iterator			Vec4fMapIter;
	typedef typename std::map< std::string, VEC4F_VEC >::const_iterator		Vec4fMapCIter;

	typedef typename std::map< std::string, VEC4H_VEC >						Vec4hMapType;
	typedef typename std::map< std::string, VEC4H_VEC >::iterator			Vec4hMapIter;
	typedef typename std::map< std::string, VEC4H_VEC >::const_iterator		Vec4hMapCIter;

	// data members
	mutable IntMapType		m_IntParams;
	mutable FloatMapType	m_FloatParams;
	mutable HalfMapType		m_HalfParams;
	mutable Vec2fMapType	m_Vec2fParams;
	mutable Vec3fMapType	m_Vec3fParams;
	mutable Vec3hMapType	m_Vec3hParams;
	mutable Vec4fMapType	m_Vec4fParams;
	mutable Vec4hMapType	m_Vec4hParams;

};

//! a VectorSet for host vectors
typedef VectorSet<	HostVector<int>,
					HostVector<float>,
					HostVector<half>,
					HostVector<Imath::V2f>,
					HostVector<Imath::V3f>,
					HostVector<Imath::V3h>,
					HostVector<Imath::V4f>,
					HostVector<Imath::V4h> >
HostVectorSet;

//! a VectorSet fpr device vectors
typedef VectorSet<	DeviceVector<int>,
					DeviceVector<float>,
					DeviceVector<half>,
					DeviceVector<Imath::V2f>,
					DeviceVector<Imath::V3f>,
					DeviceVector<Imath::V3h>,
					DeviceVector<Imath::V4f>,
					DeviceVector<Imath::V4h> >
DeviceVectorSet;




} // grind


#endif /* grind_device_vector_set_h */


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
