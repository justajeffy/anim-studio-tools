/*
 * forwardLightingTest.cpp
 *
 *  Created on: Sep 14, 2009
 *      Author: stephane.bertout
 */

#include "simpleHwShadowTest.h"

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

SimpleHwShadowTest::SimpleHwShadowTest(QWidget *parent) :
	GenericGL(parent), m_NearClipPlane(0.01f), m_FarClipPlane(5000.f)
{
	xRot = 0;//4800;
	yRot = 0;//4800;
	zRot = 0;

	clearColor = Colour(64, 64, 64, 64);

	//m_CameraFrame.SetPosition(Imath::V3f(-102.9, -77.5, -352));
	m_CameraFrame.setPosition(Imath::V3f(0, 0, 200));

	m_LightFrame.setPosition(Imath::V3f(0, 0, 100));
}

SimpleHwShadowTest::~SimpleHwShadowTest()
{
	makeCurrent();
	//glDeleteLists(object, 1);
}

QSize SimpleHwShadowTest::minimumSizeHint() const
{
	return QSize(50, 50);
}

QSize SimpleHwShadowTest::sizeHint() const
{
	return QSize(400, 400);
}

void SimpleHwShadowTest::initializeGL()
{
	m_Renderer.init();

	//********************************* TEXTURE *********************************//

	{
		//string texFileName = "data/wood.jpg"; // TOFIX !
		//string texFileName = "data/wood.jpg";
		string texFileName = "data/fromDavid/mumbleBodyCLR.tif";
		//string texFileName = "data/glr_todl_body_color_v16.tif";

		TextureLoader textureLoader(texFileName);
		textureLoader.reportStats();
		m_Texture = textureLoader.createTexture();
	}

	{
		string texFileName = "data/wood.jpg";

		TextureLoader textureLoader(texFileName);
		textureLoader.reportStats();
		m_Texture2 = textureLoader.createTexture();
	}

	m_Program = new Program("glsl/testShadow.vs.glsl",
			"glsl/testShadow.fs.glsl");

	m_ShadowUpdateProgram = new Program("glsl/testShadowUp.vs.glsl",
			"glsl/testShadowUp.fs.glsl");

	// Load our obj file
	{
		boost::shared_ptr< MeshLoader > loader ( Loader::CreateMeshLoader( "data/fromDavid/masterPenguins.obj" ) );
		loader->load();
		loader->reportStats();

		m_Mesh = loader->createMesh();
	}

	// Load our obj file
	{
		String objToLoad("data/fromMaya/cube.obj");

		boost::shared_ptr< MeshLoader > loader ( Loader::CreateMeshLoader( objToLoad ) );
		loader->load();
		loader->reportStats();

		m_Cube = loader->createMesh();
	}

	m_ModelMatrix.makeIdentity();

	// create our shadow map
	int w = 512, h = 512;
	m_ShadowRenderTarget = new RenderTarget(w, h, Texture::eRGBA, Texture::e2D,
			1, Texture::eDepth24);

	// useful..
	m_DefaultRenderTarget = new RenderTarget(m_Width, m_Height);

	// just for debugging
	// create our screen quad
	m_FSQuad = new Primitive2D(Primitive2D::eQuad);

	Colour white(255, 255, 255, 255);
	Imath::V3f pos[4] =
	{ Imath::V3f(0.6, 0.6, 0), Imath::V3f(1, 0.6, 0), Imath::V3f(1, 1, 0), Imath::V3f(0.6, 1, 0) };
	Vec4 col[4] =
	{ Vec4(1, 1, 1, 1), Vec4(1, 1, 1, 1), Vec4(1, 1, 1, 1), Vec4(1, 1, 1, 1) };
	Imath::V2f txc[4] =
	{ Imath::V2f(0, 0), Imath::V2f(1, 0), Imath::V2f(1, 1), Imath::V2f(0, 1) };
	m_FSQuad->create(pos, txc, col);

	// and full screen program
	m_FSProgram = new Program("glsl/test2D.vs.glsl", "glsl/test2D.fs.glsl");

	// and the light/shadow projection matrix
	m_ShadowProjMatrix.makeIdentity();
	makePerspective(m_ShadowProjMatrix, 45.0f, 1, 0.1, 5000);
}

void SimpleHwShadowTest::paintGL()
{
	float shadowBiasScale = -10000;

	float yCoeff = ((float) yRot) / 5760;
	float shadowBias = yCoeff;
	//SPAM( shadowBias );

	Imath::V3f lightDir;
	m_LightFrame.checkDirty();
	m_LightFrame.getForwardVector(lightDir);
	m_MoveLight = m_GlSettings->getClearInBlackConfig();

	m_ShadowViewFrame.setPosition(lightDir * 250);
	m_ShadowViewFrame.setLookAtDirection(-lightDir);

	// Model matrix
	m_ModelMatrix.makeIdentity();

	Matrix cubeMtx = m_ModelMatrix;
	cubeMtx.setScale(50.f);
	setTranslation(cubeMtx, m_ModelMatrix.translation() - Imath::V3f(0, 125, 0));

	m_ModelMatrix.setTranslation((Imath::V3f(xRot - 2880, 5520 - 2880, zRot - 2880)
			/ 2880.0f - Imath::V3f(1, 1, 1)) * Imath::V3f(10, 10, 10));

	// Update the shadow map first..
	m_ShadowRenderTarget->use();

	m_Renderer.setClearColour(Colour(255, 0, 0, 0)); // just clear in red
	m_Renderer.clear((UInt) (Renderer::eColor | Renderer::eDepth));

	if (!m_GlSettings->getWireframeConfig())
	{
		m_Renderer.enableCullFace(Renderer::eFront); // cull the front faces to prevent shadow acne artefacts

		//glEnable( GL_POLYGON_OFFSET_FILL );
		//glPolygonOffset( shadowBias * shadowBiasScale, shadowBias * shadowBiasScale );
	}

	// render the penguin from the light point of view
	m_ShadowUpdateProgram->use();
	m_ShadowUpdateProgram->setUniformMatrix("ModelMatrix", m_ModelMatrix.getValue(), false);
	m_ShadowUpdateProgram->setUniformMatrix("ViewMatrix", m_ShadowViewFrame.getMatrix().getValue(), false);
	m_ShadowUpdateProgram->setUniformMatrix("ProjMatrix", m_ShadowProjMatrix.getValue(), false);

	m_Texture->use(0, m_ShadowUpdateProgram);
	m_Mesh->use(m_ShadowUpdateProgram);
	m_Mesh->draw();

	m_Texture2->use(0, m_ShadowUpdateProgram);
	m_Cube->use(m_ShadowUpdateProgram);
	m_ShadowUpdateProgram->setUniformMatrix("ModelMatrix", cubeMtx.getValue(), false);
	m_Cube->draw();

	// go back on default FB
	m_DefaultRenderTarget->use();
	resetViewport();
	m_Renderer.disableCullFace();

	m_Renderer.setClearColour(clearColor);
	m_Renderer.clear((UInt) (Renderer::eColor | Renderer::eDepth));
	//m_Renderer.EnableWireFrame( m_GlSettings->getWireframeConfig() );

	m_Program->use();
	m_Program->setUniformMatrix("ModelMatrix", m_ModelMatrix.getValue(), false);
	m_Program->setUniformMatrix("ViewMatrix", m_CameraFrame.getMatrix().getValue(), false);
	m_Program->setUniformMatrix("ProjMatrix", m_ProjMatrix.getValue(), false);

	// for shadows
	m_Program->setUniformMatrix("ShadowViewMatrix", m_ShadowViewFrame.getMatrix().getValue(), false);
	m_Program->setUniformMatrix("ShadowProjMatrix", m_ShadowProjMatrix.getValue(), false);
	//m_Program->setUniform("ShadowBias", shadowBias * shadowBiasScale );
	m_Program->setUniform("ShadowBlurScale", shadowBias * 0.1f);

	Matrix currentViewMatrix = m_CameraFrame.getMatrix();
	Imath::V3f viewSpaceLightDir;
	currentViewMatrix.multDirMatrix(lightDir, viewSpaceLightDir);

	m_Program->setUniformVec3("LightDir", viewSpaceLightDir);

	m_Texture->use(0, m_Program);
	//m_ShadowRenderTarget->getTexture(0)->use( 0, m_Program );
	m_ShadowRenderTarget->getDepthTexture()->use(1, m_Program); // set shadow map
	m_Mesh->use(m_Program);
	m_Mesh->draw();

	m_Texture2->use(0, m_Program);
	m_Cube->use(m_Program);
	m_Program->setUniformMatrix("ModelMatrix", cubeMtx.getValue(), false);
	m_Cube->draw();

	// just for debugging..
	// display our quad
	m_FSProgram->use();
	m_FSQuad->use(m_FSProgram);
	m_ShadowRenderTarget->getTexture(0)->use(0, m_FSProgram);
	m_FSQuad->draw();
}

void SimpleHwShadowTest::resetViewport()
{
	m_DefaultRenderTarget->resize(m_Width, m_Height);
}

void SimpleHwShadowTest::resizeGL(int width, int height)
{
	m_Width = width;
	m_Height = height;
	resetViewport();

	m_ProjMatrix.makeIdentity();
	float aspect = (float) width / (float) height;
	makePerspective(m_ProjMatrix, 45.0f, aspect, m_NearClipPlane,
			m_FarClipPlane);
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
