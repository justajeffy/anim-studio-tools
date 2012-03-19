/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: http://svn/drd/apps/bee/trunk/lib/gl/Renderer.cpp $"
 * SVN_META_ID = "$Id: Renderer.cpp 91936 2011-07-19 04:39:14Z oliver.farkas $"
 */

#include "Renderer.h"
#include "glExtensions.h"
#include "Texture.h"

#include <stdio.h>
#include "../kernel/spam.h"
#include "../kernel/assert.h"

#include <math.h>
#include "../math/Imath.h"
#include <OpenColorIO/OpenColorIO.h>
namespace OCIO = OCIO_NAMESPACE;

using namespace bee;

Renderer::Renderer()
: m_CullFaceEnabled( true )
, m_CullFace( eBack )
{
	// Init ClearColor
	m_ClearColor = Colour( 0, 0, 0, 0 );
	glClearColor( m_ClearColor.r, m_ClearColor.g, m_ClearColor.b, m_ClearColor.a );
}

inline const char*
getGLVersion()
{
	return (const char*) glGetString( GL_VERSION );
}

inline const char*
getGLSLVersion()
{
	return (const char*) glGetString( GL_SHADING_LANGUAGE_VERSION );
}

void Renderer::init()
{
	std::cout << "<<< OGL Version : " << getGLVersion() << " >>>" << std::endl;
	std::cout << "<<< GLSL Version : " << getGLSLVersion() << " >>>" << std::endl;

	glEnable( GL_DEPTH_TEST );
	glDepthFunc( GL_LEQUAL );

	glEnable( GL_TEXTURE_2D );

	glClearDepth( 1.0f );
	glEnable( GL_CULL_FACE );
	glCullFace( GL_BACK );

	init_extensions();
}

void Renderer::init_extensions()
{
	bool extInited = initGLExtensions();
	Assert( extInited && "Some used extensions haven't been found..." );
}

std::string Renderer::initLut()
{
	// Step 0: Get the processor using any of the pipelines mentioned above.

	#define LUT3D_EDGE_SIZE 32

	OCIO::ConstConfigRcPtr config = OCIO::GetCurrentConfig(); // to directly use the default one matching $OCIO

	const char * device = config->getDefaultDisplay();
	const char * view = config->getDefaultView(device);
	const char * displayColorSpace = config->getDisplayColorSpaceName(device, view);

	printf("OCIO: device= %s | ", device);
	printf("view = %s | ", view);
	printf("displayColorSpace= %s \n", displayColorSpace);

	std::string g_inputColorSpace = OCIO::ROLE_SCENE_LINEAR;
	float g_exposure_fstop = 0.0f;
	int g_channelHot[4] = { 1, 1, 1, 1 }; // show rgb

	OCIO::DisplayTransformRcPtr transform = OCIO::DisplayTransform::Create();
	transform->setInputColorSpaceName( g_inputColorSpace.c_str() );
	transform->setDisplayColorSpaceName( displayColorSpace );

	// Add custom (optional) transforms for our 'canonical' display pipeline
	{
		// Add an fstop exposure control (in SCENE_LINEAR)
		float gain = powf(2.0f, g_exposure_fstop);
		const float slope3f[] = { gain, gain, gain };
		OCIO::CDLTransformRcPtr cc = OCIO::CDLTransform::Create();
		cc->setSlope(slope3f);
		transform->setLinearCC(cc);

		// Add Channel swizzling
		float lumacoef[3];
		config->getDefaultLumaCoefs(lumacoef);

		float m44[16];
		float offset[4];
		OCIO::MatrixTransform::View(m44, offset, g_channelHot, lumacoef);

		OCIO::MatrixTransformRcPtr swizzle = OCIO::MatrixTransform::Create();
		swizzle->setValue(m44, offset);
		transform->setChannelView(swizzle);
	}

	OCIO::ConstProcessorRcPtr processor = config->getProcessor(transform);
	int lut3DEdgeLen = 32;

	// Step 1: Create a GPU Shader Description
	OCIO::GpuShaderDesc shaderDesc;
	shaderDesc.setLanguage( OCIO::GPU_LANGUAGE_GLSL_1_0 );
	shaderDesc.setFunctionName( "OCIODisplay" );
	shaderDesc.setLut3DEdgeLen( lut3DEdgeLen );

	const char * gpuShaderText = processor->getGpuShaderText( shaderDesc );

	// RGB format !
	Vec3 lut3d[ lut3DEdgeLen ][ lut3DEdgeLen ][ lut3DEdgeLen ];
	memset(lut3d, sizeof(Vec3) * lut3DEdgeLen * lut3DEdgeLen * lut3DEdgeLen, 0 );
	processor->getGpuLut3D( (float *) lut3d, shaderDesc );

	m_Lut3dTexture = SharedPtr<Texture>( new Texture( lut3DEdgeLen, lut3DEdgeLen, lut3DEdgeLen, Texture::eRGB32F, Texture::e3D ) );
	m_Lut3dTexture->init( lut3d );
	// the bee texture init stuff does init this texture in a "bad" way
	// too many dependencies to update/fix it so let's just call
	// some more gl functions here..
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

	return std::string( gpuShaderText );
}

void Renderer::setClearColour( const Colour & a_Color )
{
	if ( m_ClearColor != a_Color )
	{
		glClearColor( a_Color.r, a_Color.g, a_Color.b, a_Color.a );
		m_ClearColor = a_Color;
	}
}

void Renderer::clear( UInt a_ClearFlags )
{
	glClear( 	(( a_ClearFlags & eColor ) ? ( GL_COLOR_BUFFER_BIT ) : (0))
			 | 	(( a_ClearFlags & eDepth ) ? ( GL_DEPTH_BUFFER_BIT ) : (0))
		);
}

GLuint s_CullFaceConv[] = { GL_FRONT, 			// eFront
							GL_BACK, 			// eBack
							GL_FRONT_AND_BACK, 	// eFrontAndBack
						};
void Renderer::enableCullFace( Renderer::CullFace a_CullFace )
{
	if ( !m_CullFaceEnabled )
	{
		glEnable( GL_CULL_FACE );
		m_CullFaceEnabled = true;
	}

	if ( m_CullFace != a_CullFace )
	{
		glCullFace( s_CullFaceConv[ a_CullFace ] );
		m_CullFace = a_CullFace;
	}
}

void Renderer::disableCullFace()
{
	if ( m_CullFaceEnabled )
	{
		glDisable( GL_CULL_FACE );
		m_CullFaceEnabled = false;
	}
}

void Renderer::enableWireFrame( bool a_WireFrame )
{
	if ( a_WireFrame ) 	glPolygonMode( GL_FRONT_AND_BACK, GL_LINE );
	else 				glPolygonMode( GL_FRONT_AND_BACK, GL_FILL );
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
