/*
 * forwardLightingTest.cpp
 *
 *  Created on: Sep 14, 2009
 *      Author: stephane.bertout
 */

#include "forwardLightingTest.h"

#include <QtGui>

#include <stdio.h>
#include "kernel/spam.h"
#include "kernel/string.h"
#include <boost/assert.hpp>
#include <drdDebug/log.h>

#include "gl/Mesh.h"
#include "gl/Shader.h"
#include "gl/Program.h"
#include "gl/Texture.h"
#include "gl/RenderTarget.h"
#include "io/MeshLoader.h"
#include "io/VacLoader.h"
#include "io/textureLoader.h"

using namespace std;


ForwardLightingTest::ForwardLightingTest(QWidget *parent)
    : GenericGL(parent)
    , m_NearClipPlane( 0.01f )
    , m_FarClipPlane( 5000.f )
{
    xRot = 0;//4800;
    yRot = 0;//4800;
    zRot = 0;

    clearColor = Colour(64,64,64,64);

    //m_CameraFrame.SetPosition(Imath::V3f(-102.9, -77.5, -352));
    m_CameraFrame.setPosition(Imath::V3f(0, 0, 100));

    m_LightFrame.setPosition(Imath::V3f(0, 0, 100));
}

ForwardLightingTest::~ForwardLightingTest()
{
    makeCurrent();
    //glDeleteLists(object, 1);
}

QSize ForwardLightingTest::minimumSizeHint() const
{
    return QSize(50, 50);
}

QSize ForwardLightingTest::sizeHint() const
{
    return QSize(400, 400);
}

void ForwardLightingTest::initializeGL()
{
	m_Renderer.init();

    //********************************* TEXTURE *********************************//

    {
		//string texFileName = "./../test_data/wood.jpg"; // TOFIX !
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

	m_Program = new Program( "glsl/test.vs.glsl", "glsl/test.fs.glsl" );

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

	{
		VacLoader vl( "/drd/jobs/hf2/global/rnd/testData/dmo/grind/mangle/vac/Mumble_00.h5:Hierarchy/mesh;/drd/jobs/hf2/global/rnd/testData/dmo/grind/mangle/vac/Mumble_01.h5:Hierarchy/mesh" );
		vl.open();
		vl.reportStats();
	}
//	{
//		boost::shared_ptr< MeshLoader > loader = Loader::CreateMeshLoader(
//		      "/drd/jobs/hf2/wip/depts/surface/users/romain.cote/Mumble/geo/Mumble_newAnimationNewVAC.h5:Hierarchy/MumbleROM/MumbleROMShape" );
//		loader->open();
//		loader->reportStats();
//	}
//	{
//		boost::shared_ptr< MeshLoader > loader = Loader::CreateMeshLoader(
//		      "data/fromDavid/masterPenguins.obj" );
//		loader->open();
//		loader->reportStats();
//	}




	m_ModelMatrix.makeIdentity();
}

void ForwardLightingTest::paintGL()
{
	m_MoveLight = m_GlSettings->getClearInBlackConfig();

	if ( m_GlSettings->getClearInBlackConfig() )
	{
		m_Renderer.enableCullFace( Renderer::eBack );
		m_Renderer.setClearColour( Colour(0,0,0,0) );
	}
	else
	{
		m_Renderer.disableCullFace();
		m_Renderer.setClearColour( clearColor );
	}

	m_Renderer.clear( (UInt) ( Renderer::eColor | Renderer::eDepth ) );
	m_Renderer.enableWireFrame( m_GlSettings->getWireframeConfig() );

    // Model matrix
    m_ModelMatrix.makeIdentity();
    m_ModelMatrix.setTranslation((Imath::V3f( xRot-2880, yRot-2880, zRot-2880 )/2880.0f-Imath::V3f(1,1,1))*Imath::V3f(10,10,10));

    m_Program->use();
    m_Program->setUniformMatrix("ModelMatrix", m_ModelMatrix.getValue(), false);
    m_Program->setUniformMatrix("ViewMatrix", m_CameraFrame.getMatrix().getValue(), false);
    m_Program->setUniformMatrix("ProjMatrix", m_ProjMatrix.getValue(), false);

    Imath::V3f lightDir;
    m_LightFrame.checkDirty();
    m_LightFrame.getForwardVector( lightDir );
    m_Program->setUniformVec3("LightDir", lightDir );

    m_Texture2->use( 0, m_Program );
    m_Texture2->use( 1, m_Program );
    m_Texture2->use( 2, m_Program );
    m_Texture2->use( 3, m_Program );
    m_Texture2->use( 4, m_Program );
    m_Texture2->use( 5, m_Program );
    m_Texture2->use( 6, m_Program );
    m_Texture2->use( 7, m_Program );
    m_Mesh->use( m_Program );
    m_Mesh->draw();

    // draw another penguin
    m_ModelMatrix.setTranslation((Imath::V3f( xRot+2880, yRot-2880, zRot-2880 )/2880.0f-Imath::V3f(1,1,1))*Imath::V3f(10,10,10));
    m_Program->setUniformMatrix("ModelMatrix", m_ModelMatrix.getValue(), false);
    m_Mesh->draw();
}


void ForwardLightingTest::resizeGL(int width, int height)
{
    int side = qMin(width, height);
    glViewport((width - side) / 2, (height - side) / 2, side, side);

    m_ProjMatrix.makeIdentity();
    float aspect = (float)width / (float)height;
    makePerspective( m_ProjMatrix, 45.0f, aspect, m_NearClipPlane, m_FarClipPlane);
    //SPAM(m_ProjMatrix);
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
