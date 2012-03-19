/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: $"
 * SVN_META_ID = "$Id: vector_set.cpp 45312 2010-09-09 06:11:02Z chris.cooper $"
 */

//-------------------------------------------------------------------------------------------------
#include <drdDebug/log.h>
DRD_MKLOGGER( L, "drd.grind.VectorSet" );

#include "vector_set.h"
#include "device_vector.h"
#include "host_vector.h"


using namespace grind;
using namespace drd;

#define TYPENAMES_LONG \
	typename INT_VEC, \
	typename FLOAT_VEC, \
	typename HALF_VEC, \
	typename VEC2F_VEC, \
	typename VEC3F_VEC, \
	typename VEC3H_VEC, \
	typename VEC4F_VEC, \
	typename VEC4H_VEC

#define TYPENAMES_SHORT \
		INT_VEC, \
		FLOAT_VEC, \
		HALF_VEC, \
		VEC2F_VEC, \
		VEC3F_VEC, \
		VEC3H_VEC, \
		VEC4F_VEC, \
		VEC4H_VEC

//-------------------------------------------------------------------------------------------------
#define HAS_PARAM(_fn, _map, _iter ) \
template< TYPENAMES_LONG > \
bool VectorSet< TYPENAMES_SHORT >:: \
_fn( const std::string& i_Name ) const \
{ \
	_iter it = _map.find(i_Name); \
	return it != _map.end(); \
}


HAS_PARAM( hasIntParam,		m_IntParams,	IntMapCIter );
HAS_PARAM( hasFloatParam,	m_FloatParams,	FloatMapCIter );
HAS_PARAM( hasHalfParam,	m_HalfParams,	HalfMapCIter );
HAS_PARAM( hasVec2fParam,	m_Vec2fParams,	Vec2fMapCIter );
HAS_PARAM( hasVec3fParam,	m_Vec3fParams,	Vec3fMapCIter );
HAS_PARAM( hasVec3hParam,	m_Vec3hParams,	Vec3hMapCIter );
HAS_PARAM( hasVec4fParam,	m_Vec4fParams,	Vec4fMapCIter );
HAS_PARAM( hasVec4hParam,	m_Vec4hParams,	Vec4hMapCIter );

//-------------------------------------------------------------------------------------------------
#define GET_PARAM_NAMES(_fn, _map, _iter ) \
template<	TYPENAMES_LONG > \
void VectorSet< TYPENAMES_SHORT >:: \
_fn( std::vector< std::string >& o_Result ) const \
{ \
	o_Result.clear(); \
	_iter it = _map.begin(); \
	for( ; it != _map.end(); ++it ) { \
		o_Result.push_back((*it).first); \
	} \
}


GET_PARAM_NAMES( getIntParamNames,		m_IntParams,	IntMapCIter );
GET_PARAM_NAMES( getFloatParamNames,	m_FloatParams,	FloatMapCIter );
GET_PARAM_NAMES( getHalfParamNames,		m_HalfParams,	HalfMapCIter );
GET_PARAM_NAMES( getVec2fParamNames,	m_Vec2fParams,	Vec2fMapCIter );
GET_PARAM_NAMES( getVec3fParamNames,	m_Vec3fParams,	Vec3fMapCIter );
GET_PARAM_NAMES( getVec3hParamNames,	m_Vec3hParams,	Vec3hMapCIter );
GET_PARAM_NAMES( getVec4fParamNames,	m_Vec4fParams,	Vec4fMapCIter );
GET_PARAM_NAMES( getVec4hParamNames,	m_Vec4hParams,	Vec4hMapCIter );

//-------------------------------------------------------------------------------------------------
#define GET_PARAM( _fn, _map, _type ) \
template< TYPENAMES_LONG > \
_type& VectorSet< TYPENAMES_SHORT >:: \
_fn( const std::string& i_Name ) \
{ \
	return _map[ i_Name ]; \
}

GET_PARAM( getIntParam,		m_IntParams,	INT_VEC );
GET_PARAM( getFloatParam,	m_FloatParams,	FLOAT_VEC );
GET_PARAM( getHalfParam,	m_HalfParams,	HALF_VEC );
GET_PARAM( getVec2fParam,	m_Vec2fParams,	VEC2F_VEC );
GET_PARAM( getVec3fParam,	m_Vec3fParams,	VEC3F_VEC );
GET_PARAM( getVec3hParam,	m_Vec3hParams,	VEC3H_VEC );
GET_PARAM( getVec4fParam,	m_Vec4fParams,	VEC4F_VEC );
GET_PARAM( getVec4hParam,	m_Vec4hParams,	VEC4H_VEC );

//-------------------------------------------------------------------------------------------------
#define GET_CONST_PARAM( _fn, _map, _type ) \
template< TYPENAMES_LONG > \
const _type& VectorSet< TYPENAMES_SHORT >:: \
_fn( const std::string& i_Name ) const \
{ \
	return _map[ i_Name ]; \
}

GET_CONST_PARAM( getIntParam,	m_IntParams,	INT_VEC );
GET_CONST_PARAM( getFloatParam,	m_FloatParams,	FLOAT_VEC );
GET_CONST_PARAM( getHalfParam,	m_HalfParams,	HALF_VEC );
GET_CONST_PARAM( getVec2fParam,	m_Vec2fParams,	VEC2F_VEC );
GET_CONST_PARAM( getVec3fParam,	m_Vec3fParams,	VEC3F_VEC );
GET_CONST_PARAM( getVec3hParam,	m_Vec3hParams,	VEC3H_VEC );
GET_CONST_PARAM( getVec4fParam,	m_Vec4fParams,	VEC4F_VEC );
GET_CONST_PARAM( getVec4hParam,	m_Vec4hParams,	VEC4H_VEC );

// explicit instantiations
template class VectorSet< HostVector<int>, HostVector<float>, HostVector<half>, HostVector<Imath::V2f>, HostVector<Imath::V3f>, HostVector<Imath::V3h>, HostVector<Imath::V4f>, HostVector<Imath::V4h> >;
template class VectorSet< DeviceVector<int>, DeviceVector<float>, DeviceVector<half>, DeviceVector<Imath::V2f>, DeviceVector<Imath::V3f>, DeviceVector<Imath::V3h>, DeviceVector<Imath::V4f>, DeviceVector<Imath::V4h> >;




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
