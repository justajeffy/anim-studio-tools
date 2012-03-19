#define GL_GLEXT_PROTOTYPES
#include <GL/glew.h>
#include <GL/glut.h>

#include <QApplication>
#include <QtGui>

#include <QGLWidget>
#include "window.h"
#include "glSettings.h"
#include "genericgl.h"

Window::Window( GenericGL * a_GenericGL )
:	glWidget( a_GenericGL )
{
    // Create sliders
    xSlider = createSlider();
    ySlider = createSlider();
    zSlider = createSlider();

    // Create COM stuff
    connect( xSlider, SIGNAL( valueChanged(int) ), glWidget, SLOT( setXRotation( int ) ) );
    connect( glWidget, SIGNAL( xRotationChanged(int) ), xSlider, SLOT( setValue( int ) ) );
    connect( ySlider, SIGNAL( valueChanged(int) ), glWidget, SLOT( setYRotation( int ) ) );
    connect( glWidget, SIGNAL( yRotationChanged(int) ), ySlider, SLOT( setValue( int ) ) );
    connect( zSlider, SIGNAL( valueChanged(int) ), glWidget, SLOT( setZRotation( int ) ) );
    connect( glWidget, SIGNAL( zRotationChanged(int) ), zSlider, SLOT( setValue( int ) ) );

    // Create UI
    QSplitter * splitter = new QSplitter(Qt::Horizontal, this);

    splitter->addWidget(  glWidget );
    QSplitter * rightSplitter = new QSplitter(Qt::Vertical, splitter);

    {
		QWidget *models = new QWidget (rightSplitter);
		QBoxLayout *modelsLayout = new QVBoxLayout (models);
		QLabel *ml = new QLabel ("Model List", models);
		modelsLayout->addWidget (ml);
		QListView *mv = new QListView (models);
		modelsLayout->addWidget (mv);
    }
    {
		QWidget *models = new QWidget (rightSplitter);
		QBoxLayout *modelsLayout = new QVBoxLayout (models);
		modelsLayout->addWidget(xSlider);
		modelsLayout->addWidget(ySlider);
		modelsLayout->addWidget(zSlider);
    }
    {
		QWidget *widget = new QWidget (rightSplitter);
		GLSettings *settingsLayout = new GLSettings(widget);
		//settingsLayout->setupUi(widget);

		glWidget->SetSettings( settingsLayout );
    }

    setCentralWidget(splitter);

    xSlider->setValue(0 * 16);
    ySlider->setValue(0 * 16);
    zSlider->setValue(0 * 16);

    setWindowTitle(tr("Hello GL"));
}

Window::~Window()
{
}

QSlider *Window::createSlider()
{
    QSlider *slider = new QSlider(Qt::Horizontal);
    slider->setRange(0, 360 * 16);
    slider->setSingleStep(16);
    slider->setPageStep(15 * 16);
    slider->setTickInterval(15 * 16);
    slider->setTickPosition(QSlider::TicksRight);
    return slider;
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
