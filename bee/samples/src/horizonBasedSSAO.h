#ifndef HORIZONBASEDSSAO_H
#define HORIZONBASEDSSAO_H

#include <GL/glew.h>
#include <GL/glut.h>

#include <QGLWidget>
#include <QTime>
#include "genericgl.h"
#include "glSettings.h"
#include "kernel/color.h"
#include "kernel/smartPointers.h"
#include "math/Imath.h"
#include "gl/Frame.h"
#include "gl/Renderer.h"

#include <vector>

using namespace bee;

namespace bee
{
	class Mesh;
	class Texture;
	class Program;
	class RenderTarget;
	class Primitive2D;
}

class HorizonBasedSSAO : public GenericGL
{
    Q_OBJECT

public:
    HorizonBasedSSAO(QWidget *parent = 0);
    virtual ~HorizonBasedSSAO();

    virtual QSize minimumSizeHint() const;
    virtual QSize sizeHint() const;

    //void timerEvent(QTimerEvent *) { update(); }

protected:
	void initializeGL();
	void paintGL();
	void resizeGL(int width, int height);

private:
	GLuint createTeapot(GLint grid, GLdouble scale, GLenum type);
	void normalizeAngle(int *angle);
	void resetViewport();

	int timerId;
	QTime qTime;
	int previousElapsed;

	SharedPtr< Mesh > m_Mesh;
	SharedPtr< Texture > m_Texture;
	SharedPtr< Texture > m_Texture2;
	SharedPtr< Texture > m_NoiseTexture;
	Program * m_Program;

	RenderTarget * m_RenderTarget;

	RenderTarget * m_DefaultRenderTarget;

	Primitive2D * m_FSQuad;
	Program * m_FSProgram;

	struct LightData
	{
		Imath::V3f colour;
		float animSpeed;
		float height;
	};
	std::vector< LightData > 		m_LightDataVector;

	Imath::V2f g_Resolution;
	Imath::V2f g_InvResolution;
	float g_R;
	Imath::V2f g_FocalLen;
	Imath::V2f g_InvFocalLen;
	float g_NumSteps;
	float g_NumDir;
	float  g_AngleBias;
	float  g_Attenuation;
	float  g_inv_R;
	float  g_sqr_R;

};

#endif


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
