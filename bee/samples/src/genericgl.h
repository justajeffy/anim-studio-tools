#ifndef bee_genericgl_h
#define bee_genericgl_h
#pragma once

#include "gl/Frame.h"
#include "gl/Renderer.h"

#include <QGLWidget>
#include "glSettings.h"
#include "math/Imath.h"

using namespace bee;

class GLSettings;

namespace bee
{
	class Mesh;
	class Texture;
	class Program;
	class RenderTarget;
	class Primitive2D;
}

class GenericGL : public QGLWidget
{
	Q_OBJECT
public:
	GenericGL(QWidget *parent = 0);
	GenericGL(const QGLFormat& format, QWidget *parent = 0);
	virtual ~GenericGL();

	virtual void SetSettings( GLSettings * a_GlSettings )
	{
    	m_GlSettings = a_GlSettings;
    	a_GlSettings->setUpdateCB( &UpdateCB, this );
	}
	virtual QSize minimumSizeHint() const;
	virtual QSize sizeHint() const;

	static void UpdateCB( void * a_UserData );
	void stuff();

	// COM Stuff
public slots:
	virtual void setXRotation(int angle);
	virtual void setYRotation(int angle);
	virtual void setZRotation(int angle);

signals:
	virtual void xRotationChanged(int angle);
	virtual void yRotationChanged(int angle);
	virtual void zRotationChanged(int angle);
	// end COM Stuff

protected:
	virtual void initializeGL();
	virtual void paintGL();
	virtual void resizeGL(int width, int height);

	virtual void mousePressEvent(QMouseEvent *event);
	virtual void mouseMoveEvent(QMouseEvent *event);

	virtual void keyPressEvent(QKeyEvent *);
	virtual void keyReleaseEvent(QKeyEvent *);

	const GLSettings * m_GlSettings;
	Renderer m_Renderer;

	Frame m_CameraFrame;

	Matrix m_ModelMatrix;
	Matrix m_ProjMatrix;

	bool m_MoveLight;
	Frame m_LightFrame;

	int xRot;
	int yRot;
	int zRot;
	QPoint lastPos;

	Int m_Width, m_Height;
	Colour clearColor;
	float m_NearClipPlane, m_FarClipPlane;
};

#endif // bee_genericgl_h


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
