/*
 * simpleToonShadertest.cpp
 *
 *  Created on: Sep 14, 2009
 *      Author: stephane.bertout
 */

#include "simpleToonShaderTest.h"

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

SimpleToonShaderTest::SimpleToonShaderTest(QWidget *parent)
    : GenericGL(parent)
{
    xRot = 0;//4800;
    yRot = 0;//4800;
    zRot = 0;

    clearColor = Colour( 255, 255, 255, 255 );

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

    timerId = startTimer(0);
}

SimpleToonShaderTest::~SimpleToonShaderTest()
{
    makeCurrent();
    //glDeleteLists(object, 1);
}

QSize SimpleToonShaderTest::minimumSizeHint() const
{
    return QSize(50, 50);
}

QSize SimpleToonShaderTest::sizeHint() const
{
    return QSize(400, 400);
}

void SimpleToonShaderTest::initializeGL()
{
//	makeCurrent();
	m_Renderer.init();

    //********************************* TEXTURE *********************************//

    {
		//string texFileName = "data/wood.jpg"; // TOFIX !
		//string texFileName = "data/wood.jpg";
		string texFileName = "data/fromDavid/mumbleBodyCLRtest.tga";
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

    string tamTexFileNames[tamTexCount] = { "data/fromInternet/imgA.jpg",
											"data/fromInternet/imgB.jpg",
											"data/fromInternet/imgC.jpg",
											"data/fromInternet/imgD.jpg",
											"data/fromInternet/imgE.jpg",
											"data/fromInternet/imgF.jpg"};
    for ( int iTex=0; iTex<tamTexCount; ++iTex )
    {
		TextureLoader textureLoader( tamTexFileNames[ iTex ] );
		textureLoader.reportStats();
		m_HatchTexture[ iTex ] = textureLoader.createTexture();
    }

	m_Program = new Program( "glsl/testGbuf.vs.glsl", "glsl/testGbuf4NPR.fs.glsl" );

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
	int w = 640, h = 480;
	//int w = 128, h = 128;
	m_RenderTarget = new RenderTarget( w, h, Texture::eRGBA32F, Texture::e2D, 4, Texture::eDepth24 );

	// create our full screen quad
	m_FSQuad = new Primitive2D( Primitive2D::eQuad );

	Vec4 white(1,1,1,1);
	Imath::V3f pos[4] = { Imath::V3f(-1,-1,0), Imath::V3f(1,-1,0), Imath::V3f(1,1,0), Imath::V3f(-1,1,0) };
	Vec4 col[4] = { white, white, white, white };
	Imath::V2f txc[4] = { Imath::V2f(0, 0), Imath::V2f(1, 0), Imath::V2f(1, 1), Imath::V2f(0, 1) };
	m_FSQuad->create( pos, txc, col );

	// and full screen program
	m_FSProgram = new Program( "glsl/testDefLit.vs.glsl", "glsl/testNPR.fs.glsl" );

	m_DefaultRenderTarget = new RenderTarget( m_Width, m_Height );

	//qTime.start();
	anim = 0.f;
}

void SimpleToonShaderTest::paintGL()
{
	anim += 0.01f;
//	makeCurrent();

	m_RenderTarget->use();

	m_MoveLight = m_GlSettings->getClearInBlackConfig();

	/*if ( m_GlSettings->getClearInBlackConfig() )
	{
		m_Renderer.EnableCullFace( Renderer::eBack );
		m_Renderer.SetClearColor( Color(0,0,0,0) );
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

    m_Texture->use( 0, m_Program );
    m_Mesh->use( m_Program );
    m_Mesh->draw();

    // one more penguin
    //m_ModelMatrix.setTranslation(Imath::V3f(50,-50,0));
    //m_Program->setUniformMatrix("ModelMatrix", m_ModelMatrix.getValue(), false);
    //m_Mesh->draw();

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

    for ( int iTex=0; iTex<tamTexCount; ++iTex )
    {
    	m_HatchTexture[iTex]->use( 3 + iTex, m_FSProgram );
    }

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

void SimpleToonShaderTest::resetViewport()
{
    m_DefaultRenderTarget->resize( m_Width, m_Height );
}

void SimpleToonShaderTest::resizeGL(int width, int height)
{
	m_Width = width;
	m_Height = height;
	resetViewport();

    m_ProjMatrix.makeIdentity();
    float aspect = (float)m_Width / (float)m_Height;
    makePerspective( m_ProjMatrix, 45.0f, aspect, m_NearClipPlane, m_FarClipPlane);
}

void SimpleToonShaderTest::normalizeAngle(int *angle)
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
