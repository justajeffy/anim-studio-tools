#include "glSettings.h"

GLSettings::GLSettings(QWidget *parent)
	: m_UpdateCB( NULL )
	, m_UpdatePtr( NULL )
 {
     setupUi(parent);

     ui_Wireframe->setChecked(false);
     m_WireframeConfig = false;

     clearInBlack->setChecked(false);
     m_ClearInBlackConfig = false;

     connect(clearInBlack, SIGNAL(clicked()), this, SLOT(OnClearInBlackButtonClicked()));
     connect(ui_Wireframe, SIGNAL(clicked()), this, SLOT(OnWireframeButtonClicked()));
 }

void GLSettings::onWireframeButtonClicked()
{
	m_WireframeConfig = ui_Wireframe->isChecked();
	callUpdateCB();
}

void GLSettings::onClearInBlackButtonClicked()
 {
	m_ClearInBlackConfig = clearInBlack->isChecked();
	callUpdateCB();

     /*if (nameLineEdit->text().isEmpty())
         (void) QMessageBox::information(this, tr("No Image Name"),
             tr("Please supply a name for the image."), QMessageBox::Cancel);
     else
         accept();*/
 }

void GLSettings::callUpdateCB()
{
	(*m_UpdateCB)( m_UpdatePtr );
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
