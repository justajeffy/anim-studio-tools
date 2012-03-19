
#include "horizonBasedSSAO.h"

#include <QtGui>

#include <stdio.h>
#include "kernel/spam.h"
#include "kernel/string.h"
#include <boost/assert.hpp>

#include "gl/Mesh.h"
#include "gl/Shader.h"
#include "gl/Primitive2D.h"
#include "gl/Program.h"
#include "gl/Texture.h"
#include "gl/RenderTarget.h"
#include "io/MeshLoader.h"
#include "io/textureLoader.h"

using namespace std;

HorizonBasedSSAO::HorizonBasedSSAO(QWidget *parent)
    : GenericGL(parent)
{
    clearColor = Colour( 64, 64, 64, 64 );

    m_CameraFrame.setPosition( Imath::V3f( 0, 0, 150 ) );
    m_LightFrame.setPosition( Imath::V3f( 0, 0, 0 ) );

    // create our vector of lights
    UInt maxLightCount = 1;
    m_LightDataVector.reserve( maxLightCount );
    for (UInt iLight = 0; iLight < maxLightCount; ++iLight)
    {
    	LightData LD;
    	LD.colour = Imath::V3f( float( qrand() % 255 ) / 255.0f, float( qrand() % 255 ) / 255.0f, float( qrand() % 255 ) / 255.0f );
    	LD.animSpeed = 1 + LD.colour.x * 10;
    	LD.height = 80 * ( float( qrand() % 100 ) * 0.01f - 0.5f);

    	SPAM(LD.colour);
    	SPAM(LD.animSpeed);
    	SPAM(LD.height);
    	m_LightDataVector.push_back(LD);
    }

    //timerId = startTimer(0);


	g_R = 0.01;

	g_NumSteps = 4;//8;
	g_NumDir = 8;//16;
	g_AngleBias = 30;
	g_Attenuation = 0.1;
}

HorizonBasedSSAO::~HorizonBasedSSAO()
{
    makeCurrent();
    //glDeleteLists(object, 1);
}

QSize HorizonBasedSSAO::minimumSizeHint() const
{
    return QSize(50, 50);
}

QSize HorizonBasedSSAO::sizeHint() const
{
    return QSize(400, 400);
}

void HorizonBasedSSAO::initializeGL()
{
//	makeCurrent();
	m_Renderer.init();

    //********************************* TEXTURE *********************************//

    {
		//string texFileName = "data/wood.jpg"; // TOFIX !
		//string texFileName = "data/wood.jpg";
		string texFileName = "data/fromDavid/mumbleBodyCLR.tif";
		//string texFileName = "data/glr_todl_body_color_v16.tif";

		TextureLoader textureLoader( texFileName );
		textureLoader.reportStats();
		m_Texture = textureLoader.createTexture();
    }

    {
		string texFileName = "data/wood.jpg";

		TextureLoader textureLoader( texFileName );
		textureLoader.reportStats();
		m_Texture2 = textureLoader.createTexture();
    }

    {
		string texFileName = "data/noise.jpg";

		TextureLoader textureLoader( texFileName );
		textureLoader.reportStats();
		m_NoiseTexture = textureLoader.createTexture();
    }

	m_Program = new Program( "glsl/testGbufSSAO.vs.glsl", "glsl/testGbufSSAO.fs.glsl" );

	// Load our obj file
	//string objToLoad( "data/Lamp_Table.obj" );
	//string objToLoad( "data/fromMaya/cubeLO.obj" );
	String objToLoad( "data/fromDavid/masterPenguins.obj" );
	//string objToLoad( "data/test.obj" );
	//string objToLoad( "data/fromDavid/penguin.obj" );
	//string objToLoad( "data/fromDavid/penguinLfoot.mtl.obj" );
	boost::shared_ptr< MeshLoader > loader ( Loader::CreateMeshLoader( objToLoad ) );
	loader->load();
	loader->reportStats();

	m_Mesh = loader->createMesh();
	m_ModelMatrix.makeIdentity();

	// create our render target !
	int w = 1280, h = 768;
	m_RenderTarget = new RenderTarget( w, h, Texture::eRGBA32F, Texture::e2D, 4, Texture::eDepth24 );

    g_Resolution = Imath::V2f( (float) w, (float) h );
	g_InvResolution = Imath::V2f( 1/g_Resolution.x, 1/g_Resolution.y );

	// create our full screen quad
	m_FSQuad = new Primitive2D( Primitive2D::eQuad );

	Vec4 white(1,1,1,1);
	Imath::V3f pos[4] = { Imath::V3f(-1,-1,0), Imath::V3f(1,-1,0), Imath::V3f(1,1,0), Imath::V3f(-1,1,0) };
	Vec4 col[4] = { white, white, white, white };
	Imath::V2f txc[4] = { Imath::V2f(0, 0), Imath::V2f(1, 0), Imath::V2f(1, 1), Imath::V2f(0, 1) };
	m_FSQuad->create( pos, txc, col );

	// and full screen program
	m_FSProgram = new Program( "glsl/testSSAO.vs.glsl", "glsl/testSSAO.fs.glsl" );

	m_DefaultRenderTarget = new RenderTarget( m_Width, m_Height );

	qTime.start();

	setXRotation(2800);//4800;
	setYRotation(4200);//4800;
	setZRotation(0);

}
//float anim = 0.f;

void HorizonBasedSSAO::paintGL()
{
	//anim += 0.01f;
	float anim = 0;
//	makeCurrent();

	g_R = ((float)xRot) / 5760;
	g_AngleBias = ((float)yRot) / 5760;
	g_Attenuation  = ((float)zRot) / 5760;

	g_R *= 50;
	g_AngleBias *= 60;
	g_Attenuation *= 10;

	m_RenderTarget->use();

	m_MoveLight = m_GlSettings->getClearInBlackConfig();

	/*if ( m_GlSettings->getClearInBlackConfig() )
	{
		m_Renderer.enableCullFace( Renderer::eBack );
		m_Renderer.setClearColour( Color(0,0,0,0) );
	}
	else*/
	{
		m_Renderer.disableCullFace();
		m_Renderer.setClearColour( clearColor );
	}

	m_Renderer.clear( (UInt) ( Renderer::eColor | Renderer::eDepth ) );
	m_Renderer.enableWireFrame( m_GlSettings->getWireframeConfig() );

    // Model matrix
    m_ModelMatrix.makeIdentity();
    //m_ModelMatrix.setTranslation((Imath::V3f( xRot-2880, yRot-2880, zRot-2880 )/2880.0f-Imath::V3f(1,1,1))*Imath::V3f(10,10,10));
    m_ModelMatrix.setTranslation(Imath::V3f(0,-50,0));

    Matrix currentViewMatrix = m_CameraFrame.getMatrix();
    Matrix currentInvViewMatrix = m_CameraFrame.getMatrix().inverse();

    m_Program->use();
    m_Program->setUniformMatrix("ModelMatrix", m_ModelMatrix.getValue(), false);
    m_Program->setUniformMatrix("ViewMatrix", currentViewMatrix.getValue(), false);
    m_Program->setUniformMatrix("ProjMatrix", m_ProjMatrix.getValue(), false);

    m_Texture2->use( 0, m_Program );
    //m_NoiseTexture->use( 0, m_Program );
    m_Mesh->use( m_Program );
    m_Mesh->draw();

    // draw another penguin with the rendertarget bound => should be red
    /*m_ModelMatrix.setTranslation((Imath::V3f( xRot+2880, yRot-2880, zRot-2880 )/2880.0f-Imath::V3f(1,1,1))*Imath::V3f(10,10,10));
    m_Program->setUniformMatrix("ModelMatrix", m_ModelMatrix.getValue(), false);
    m_RenderTarget->getTexture(0)->use( 0, m_Program );
    m_Mesh->draw();*/

    m_DefaultRenderTarget->use();

	m_Renderer.setClearColour( clearColor );
	m_Renderer.clear( (UInt) ( Renderer::eColor | Renderer::eDepth ) );

    // display fullscreen quad
    m_FSProgram->use();
    m_FSQuad->use( m_FSProgram );

    // set GBuffer textures
    m_RenderTarget->getTexture( 0 )->use( 0, m_FSProgram );
    m_RenderTarget->getTexture( 1 )->use( 1, m_FSProgram, true );
    m_RenderTarget->getTexture( 2 )->use( 2, m_FSProgram );
    m_NoiseTexture->use( 3, m_FSProgram );


    m_FSProgram->setUniformVec2( "g_Resolution", g_Resolution );
    m_FSProgram->setUniformVec2( "g_InvResolution", g_InvResolution );

	g_inv_R = 1 / g_R;
	g_sqr_R = g_R * g_R;

    m_FSProgram->setUniform( "g_R", g_R );
    SPAM(g_R);
    m_FSProgram->setUniformVec2( "g_FocalLen", g_FocalLen );
    SPAM(g_FocalLen);
    m_FSProgram->setUniformVec2( "g_FocalLen", g_InvFocalLen );
    SPAM(g_InvFocalLen);
    m_FSProgram->setUniform( "g_NumSteps", g_NumSteps );
    SPAM(g_NumSteps);
    m_FSProgram->setUniform( "g_NumDir", g_NumDir );
    SPAM(g_NumDir);
    m_FSProgram->setUniform( "g_AngleBias", (float) (g_AngleBias * M_PI / 180.0f) ); // in radians
    SPAM(g_AngleBias);
    m_FSProgram->setUniform( "g_Attenuation", g_Attenuation );
    SPAM(g_Attenuation);
    m_FSProgram->setUniform( "g_inv_R", g_inv_R );
    SPAM(g_inv_R);
    m_FSProgram->setUniform( "g_sqr_R", g_sqr_R );
    SPAM(g_sqr_R);

    float cameraRange = m_FarClipPlane - m_NearClipPlane;
    m_FSProgram->setUniform( "g_cameraRange", cameraRange );


    // set some lights
    // not very efficient way (one full screen quad for each light) but it's just an example... ;)
	float lightDistToOrigin = 20;
	float lightRange = 20;

    for ( UInt i = 0; i < m_LightDataVector.size(); ++i )
    {
		Imath::V3f viewSpaceLightPos, lightPos;
		Float cosA = cosf(m_LightDataVector[i].animSpeed * anim);
		Float sinA = sinf(m_LightDataVector[i].animSpeed * anim);
		lightPos = Imath::V3f( lightDistToOrigin * cosA, m_LightDataVector[i].height, lightDistToOrigin * sinA );
		currentViewMatrix.multVecMatrix( lightPos, viewSpaceLightPos );
		Imath::V3f lightColor = m_LightDataVector[i].colour;
		m_FSProgram->setUniformVec4("Light0PositionAttenuation",
			Vec4( viewSpaceLightPos.x, viewSpaceLightPos.y, viewSpaceLightPos.z, 1 / lightRange) );
		m_FSProgram->setUniformVec3("Light0Color", lightColor );
		m_FSQuad->draw();

		if (i == 0)
		{
		    // additive mode for the other ones
		    glEnable( GL_BLEND );
		    glBlendFunc( GL_ONE, GL_ONE );
		}
    }

    glDisable( GL_BLEND );

    //doneCurrent();
    int elapsedMilliseconds = qTime.elapsed();
    //cout << "FPS = " << ( elapsedMilliseconds - previousElapsed ) << endl;
    previousElapsed = elapsedMilliseconds;
}

void HorizonBasedSSAO::resetViewport()
{
    m_DefaultRenderTarget->resize( m_Width, m_Height );

//    g_Resolution = Imath::V2f( (float) m_Width, (float) m_Height );
//	g_InvResolution = Imath::V2f( 1/g_Resolution.x, 1/g_Resolution.y );
}

void HorizonBasedSSAO::resizeGL(int width, int height)
{
	m_Width = width;
	m_Height = height;
	resetViewport();

	float fovy = 45.0f;
	float aspect = (float)m_Width / (float)m_Height;

	g_FocalLen[0] = 1.0f / tanf(fovy * 0.5f) * aspect;
    g_FocalLen[1] = 1.0f / tanf(fovy * 0.5f);

	g_InvFocalLen[0] = 1.0f / g_FocalLen[0];
    g_InvFocalLen[1] = 1.0f / g_FocalLen[1];

    m_ProjMatrix.makeIdentity();
    makePerspective( m_ProjMatrix, fovy, aspect, m_NearClipPlane, m_FarClipPlane);
}

void HorizonBasedSSAO::normalizeAngle(int *angle)
{
    while (*angle < 0)
        *angle += 360 * 16;
    while (*angle > 360 * 16)
        *angle -= 360 * 16;
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
