#include <QApplication>
#include <QtGui>

#include "window.h"
#include "glSettings.h"
#include "deferredLightingTest.h"
#include "forwardLightingTest.h"
#include "horizonBasedSSAO.h"
#include "simpleHwShadowTest.h"
#include "simpleRenderTargetTest.h"
#include "simpleToonShaderTest.h"

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    //GenericGL * widget = new DeferredLightingTest();
    GenericGL * widget = new ForwardLightingTest();
    //GenericGL * widget = new HorizonBasedSSAO(); // still WIP
    //GenericGL * widget = new SimpleHwShadowTest();
    //GenericGL * widget = new SimpleRenderTargetTest();
    //GenericGL * widget = new SimpleToonShaderTest();

    Window window( widget );
    window.show();
    return app.exec();
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
