
#include "genericgl.h"

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
#include "io/textureLoader.h"

using namespace bee;

GenericGL::GenericGL(QWidget *parent)
:	QGLWidget( parent )
,	m_GlSettings( NULL )
,	m_MoveLight( false )
,	m_NearClipPlane( 0.01f )
,	m_FarClipPlane( 5000.f )
{
}

GenericGL::GenericGL(const QGLFormat& format, QWidget *parent)
:	QGLWidget( format, parent )
,	m_GlSettings( NULL )
,	m_MoveLight( false )
,	m_NearClipPlane( 0.01f )
,	m_FarClipPlane( 5000.f )
{
}

GenericGL::~GenericGL()
{

}

QSize GenericGL::minimumSizeHint() const
{
    return QSize(50, 50);
}

QSize GenericGL::sizeHint() const
{
    return QSize(m_Width, m_Height);
}

void GenericGL::setXRotation(int angle)
{
    if (angle != xRot) {
        xRot = angle;
        updateGL();
    }
}

void GenericGL::setYRotation(int angle)
{
    if (angle != yRot) {
        yRot = angle;
        updateGL();
    }
}

void GenericGL::setZRotation(int angle)
{
    if (angle != zRot) {
        zRot = angle;
        updateGL();
    }
}

void GenericGL::initializeGL()
{
}

void GenericGL::UpdateCB( void * a_UserData )
{
	((GenericGL *) a_UserData)->updateGL();
}

void GenericGL::paintGL()
{
}

void GenericGL::resizeGL(int width, int height)
{
	m_Width = width;
	m_Height = height;

    m_ProjMatrix.makeIdentity();
    float aspect = (float)m_Width / (float)m_Height;
    makePerspective( m_ProjMatrix, 45.0f, aspect, m_NearClipPlane, m_FarClipPlane);
}

void GenericGL::mousePressEvent(QMouseEvent *event)
{
    lastPos = event->pos();
}

void GenericGL::mouseMoveEvent(QMouseEvent *event)
{
    int dx = event->x() - lastPos.x();
    int dy = event->y() - lastPos.y();

    //todo: change names !
    switch ( event->buttons() )
    {
    case Qt::LeftButton:
    	if (m_MoveLight) 	m_LightFrame.rotate(-dy, -dx);
    	else				m_CameraFrame.rotate(-dy, -dx);
    	break;
    case Qt::MidButton:
		if (m_MoveLight)	m_LightFrame.translate(dy, -dx, 0);
    	else				m_CameraFrame.translate(dy, -dx, 0);
    	break;
    case Qt::RightButton:
		if (m_MoveLight)	m_LightFrame.truck( dy );
    	else				m_CameraFrame.truck( dy );
    	break;
    }

    lastPos = event->pos();

    updateGL();
}

void GenericGL::keyPressEvent(QKeyEvent * event)
{
	SPAM(event->key());
	if (event->key() == 'l') m_MoveLight = true;
	updateGL();
}

void GenericGL::keyReleaseEvent(QKeyEvent * event)
{
	SPAM(event->key());
	if (event->key() == 'l') m_MoveLight = false;
	updateGL();
}


void GenericGL::stuff()
{
	int i = 0;
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
