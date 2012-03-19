
#include <drdDebug/log.h>
DRD_MKLOGGER( L, "drd.grind.bindings.python" );

#include <boost/python.hpp>
#include <boost/python/object.hpp>
#include <boost/python/list.hpp>
#include <boost/python/dict.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

#include "grind/cube.h"
#include "grind/dump_info.h"
#include "grind/context.h"
#include "grind/texture.h"
#include "grind/program.h"
#include "grind/mesh.h"
#include "grind/mesh_subdivide.h"
#include "grind/pbo.h"
#include "grind/line_soup.h"
#include "grind/particle_system.h"
#include "grind/random.h"
#include "grind/guide_set.h"
#include "grind/device_vector_algorithms.h"
#include "grind/cuda_manager.h"

#include <grind/utils.h>

#include <bee/io/TextureTools.h>

#include <string>
#include <vector>
#include <iostream>

#include <OpenEXR/ImathMatrix.h>
#include <rx.h>

using namespace grind;
using namespace boost::python;

//-------------------------------------------------------------------------------------------------
// couldn't get bindings to work directly on these
inline float gpuMemAvailable() { return ContextInfo::instance().getGpuMemAvailable(); }
inline unsigned int gpuFreeMem() { return ContextInfo::instance().getFreeMem(); }
inline unsigned int gpuTotalMem() { return ContextInfo::instance().getTotalMem(); }

//! cleanup problematic static data whenever required (eg before grind is unloaded)
extern "C"
void cleanupGrind()
{
	DRD_LOG_INFO( L, "cleanupGrind(): clearing static data" );
//	grind::ContextInfo::destroy();
	grind::DeviceRandom::destroy();
}

//-------------------------------------------------------------------------------------------------
// just for testing, allow throwing of an error
inline void throwAnError( const std::string& err )
{
	throw std::runtime_error( err );
}

//----------------------------------------------------------------------------
/**
 * get the current transform
 */
bool get_current_rx_transform
(
	float const a_time
,	Imath::Matrix44< float >& o_matrix
)
{
	RtMatrix temp_matrix;
	bool success = ( 0 == RxTransform( "object", "world", a_time, temp_matrix ) );
	if( success )
	{
		// copy over the data
		for( size_t col = 0; col < 4; ++col )
		{
			for( size_t row = 0; row < 4; ++row )
			{
				o_matrix[ row ][ col ] = temp_matrix[ row ][ col ];
			}
		}

		return true;
	}

	return false;
}

//----------------------------------------------------------------------------
/**
 * debug print the current ri transform
 */
void print_ri_transform( float const time )
{
	if( hasRX() )
	{
		Imath::Matrix44< float > curr_transform;
		get_current_rx_transform( time, curr_transform );
		DRD_LOG_VERBATIM( L, "transform: " << curr_transform );
	}
	else
	{
		DRD_LOG_VERBATIM( L, "RX context invalid, unable to call RenderMan RxTransform function" );
	}
}

template<typename T> const T copyObject(const T& v) { return v; }


Imath::V3f & getMin( BBox & self )
{
	return self.GetBox().min;
}

void setMin( BBox & self, const Imath::V3f & value )
{
	self.GetBox().min = value;
}

Imath::V3f & getMax( BBox & self )
{
	return self.GetBox().max;
}

void setMax( BBox & self, const Imath::V3f & value )
{
	self.GetBox().max = value;
}

Imath::V3f size( const BBox & self )
{
	return self.GetBox().size();
}

void extendBy( BBox & self, const BBox & other )
{
	self.GetBox().extendBy( other.GetBox() );
}

//-------------------------------------------------------------------------------------------------
BOOST_PYTHON_MODULE(_grind)
{
	DRD_LOG_DEBUG( L, "Python Bindings" );

	// [aj] todo move these into a generic containers pybind lib!
	////////////////////
	class_< std::vector< std::string > >( "VectorString" )
		.def( vector_indexing_suite< std::vector< std::string > >() )
	;

	class_< std::vector< float > >( "VectorFloat" )
			.def( vector_indexing_suite< std::vector< float > >() )
	;

	class_< std::vector< Imath::V3f > >( "VectorVec3f" )
			.def( vector_indexing_suite< std::vector< Imath::V3f > >() )
	;
	////////////////////


	def( "SaveGLScreenShot", bee::SaveGLScreenShot );

	def( "info", dumpInfo );
	def( "cleanup", cleanupGrind );
	def( "gpu_info", dumpGPUInfo );

	def( "has_gpu", hasGPU );
	def( "has_emulation", hasEmulation );
	def( "has_opengl", hasOpenGL );
	def( "has_rx", hasRX );
	def( "gpu_mem_available", gpuMemAvailable );
	def( "gpu_free_mem", gpuFreeMem );
	def( "gpu_total_mem", gpuTotalMem );

	def( "throw", throwAnError );

	class_< PBO >( "PBO" )
		.def( "read", &PBO::read )
		.def( "sample", &PBO::sample )
	;

	// device vector algorithms
	def( "perturb", perturb );
	def( "remap", remap );
	//def( "set_all_elements", setAllElements );

	class_< DeviceVectorHandle<float> >( "FloatArray" )
	;

	class_< Renderable >( "Renderable" )
		.def( "render", &Renderable::render )
		.def( "get_bounds", &Renderable::getBounds )
	;

	class_< BBox, bases<Renderable> >( "BBox" )
		.add_property("min", make_function(getMin, return_value_policy<reference_existing_object>()), setMin )
		.add_property("max", make_function(getMax, return_value_policy<reference_existing_object>()), setMax )
		.def( "center", &BBox::center )
		.def( "size", &size )
		.def( "pad", &BBox::pad )
		.def( "set_colour_index", &BBox::setColourIndex )
		.def( "set_colour", &BBox::setColour )
		.def( "is_empty", &BBox::isEmpty )
		.def( "extend_by", extendBy )
	;

	class_< Cube, bases<Renderable> >( "Cube" )
	;

	class_< ParticleSystem, bases<Renderable>  >( "ParticleSystem" )
		.def( "read", &ParticleSystem::read )
		.def( "get_gpu_size_of", &ParticleSystem::getGpuSizeOf )
		.def( "get_point_count", &ParticleSystem::getPointCount )
		.def( "get_width", &ParticleSystem::getWidth )
		.def( "get_height", &ParticleSystem::getHeight )
		.def( "get_real_point_count", &ParticleSystem::getRealPointCount )
		.def( "set_current_user_data_index", &ParticleSystem::setCurrentUserDataIndex )
		.def( "get_current_user_data_name", &ParticleSystem::getCurrentUserDataName )
		.def( "get_current_user_data_type", &ParticleSystem::getCurrentUserDataType )
		.def( "get_current_user_data_size", &ParticleSystem::getCurrentUserDataSize )
		.def( "get_user_data_count", &ParticleSystem::getUserDataCount )
		.def( "get_user_data_name", &ParticleSystem::getUserDataName )
		.def( "get_user_data_type", &ParticleSystem::getUserDataType )

		.def( "get_bake_cam_position", &ParticleSystem::getBakeCamPosition )
		.def( "get_bake_cam_look_at", &ParticleSystem::getBakeCamLookAt )
		.def( "get_bake_cam_up", &ParticleSystem::getBakeCamUp )
		.def( "get_bake_width", &ParticleSystem::getBakeWidth )
		.def( "get_bake_height", &ParticleSystem::getBakeHeight )
		.def( "get_bake_aspect", &ParticleSystem::getBakeAspect )
		.def( "get_view_proj_matrix", &ParticleSystem::getViewProjMatrix )
		.def( "get_inv_view_proj_matrix", &ParticleSystem::getInvViewProjMatrix )
	;

	class_< Program >( "Program" )
		.def( "read", (void ( ::Program::* )( const std::string&, const std::string& ) )( &::Program::read ), ( arg("i_VertexShaderPath"), arg("i_FragmentShaderPath") ) )
		.def( "read", (void ( ::Program::* )( const std::string&, const std::string&, const std::string&, unsigned int, unsigned int ) )( &::Program::read )
					, ( arg("i_VertexShaderPath"), arg("i_FragmentShaderPath"), arg("i_GeometryShaderPath"), arg("i_GeomInType"), arg("i_GeomOutType") ) )
		.def( "use", &Program::use )
		.def( "un_use", &Program::unUse )
	;

	class_< Texture >( "Texture" )
		.def( "read", &Texture::read )
		.def( "use", &Texture::use )
		.def( "un_use", &Texture::unUse )
	;

	class_< DeviceMesh, bases<Renderable> >( "DeviceMesh" )
		.def( "read", &DeviceMesh::read )
		.def( "n_verts", &DeviceMesh::nVerts )
		.def( "set_frame", &DeviceMesh::setFrame )
		.def( "set_P", &DeviceMesh::setP )
		.def( "set_N", &DeviceMesh::setN )
		.def( "set_points_and_normals_changed", &DeviceMesh::setPointsAndNormalsChanged )
		.def( "set_display_normals", &DeviceMesh::setDisplayNormals )
		.def( "get_display_normals", &DeviceMesh::getDisplayNormals )
        .def( "draw_normals", &DeviceMesh::drawNormals )
        .def( "info", &DeviceMesh::info )
        .def( "set_to_static_pose", &DeviceMesh::setToStaticPose )
        .def( "set_auto_apply_transforms", &DeviceMesh::setAutoApplyTransforms )
		.def( "get_auto_apply_transforms", &DeviceMesh::getAutoApplyTransforms )
		.def( "set_reference_frame", &DeviceMesh::setReferenceFrame )
		.def( "save_data", &DeviceMesh::saveData )
		.def( "__copy__", copyObject<DeviceMesh> )
		;

	;
#ifndef __SUPPORT_SM_10__
	class_< MeshSubdivide >( "MeshSubdivide" )
		.def( "process", &MeshSubdivide::process )
		.def( "process", &MeshSubdivide::processGuides )
		.def( "set_iterations", &MeshSubdivide::setIterations )
		;
#endif

#if 0
	enum_<LineSoup::CurveTypeGL>("CurveTypeGL")
		.value("LINESOUP_GL_LINES", LineSoup::LINESOUP_GL_LINES)
		.value("LINESOUP_GL_QUADS", LineSoup::LINESOUP_GL_QUADS)
	;
#endif

	class_< LineSoup, bases<Renderable> >( "LineSoup" )
		.def( "test_setup", &LineSoup::testSetup )
		.def( "set_line_width", &LineSoup::setConstantLineWidth )
		.def( "list_params", &LineSoup::listParams )
		.def( "set_param", &LineSoup::setParam )
		.def( "set_display_normals", &LineSoup::setDisplayNormals )
		.def( "get_display_normals", &LineSoup::getDisplayNormals )
		.def( "get_data", (void ( ::grind::LineSoup::* ) ( const std::string&, std::vector< Imath::V3f >& ) ) (&::grind::LineSoup::getData) )
		.def( "set_data", (void ( ::grind::LineSoup::* ) ( const std::string&, std::vector< Imath::V3f >& ) ) (&::grind::LineSoup::setData) )
		.def( "info", &LineSoup::info )
	;

	class_< GuideSet, bases<Renderable> >( "GuideSet" )
		.def( "init", &GuideSet::init )
		.def( "read", &GuideSet::read )
		.def( "update", &GuideSet::update )
		.def( "set_frame", &GuideSet::setFrame )
		.def( "surface_normal_groom", &GuideSet::surfaceNormalGroom )
		.def( "randomize", &GuideSet::randomize )
		.def( "get_cv_count", &GuideSet::getNCVs )
		.def( "get_data", (void ( ::grind::GuideSet::* ) ( const std::string&, std::vector< Imath::V3f >& ) ) (&::grind::GuideSet::getData) )
		.def( "set_data", (void ( ::grind::GuideSet::* ) ( const std::string&, std::vector< Imath::V3f >& ) ) (&::grind::GuideSet::setData) )
		.def( "get_max_guide_length", &GuideSet::getMaxGuideLength )
		.def( "__copy__", copyObject<GuideSet> )
	;

	//! print the contents of an ri transform to std::cout
	def( "print_ri_transform", print_ri_transform );
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
