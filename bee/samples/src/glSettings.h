#ifndef GLSETTINGS_H
#define GLSETTINGS_H

#include "ui_settings.h"

typedef void (*UpdateCB)(void *);

class GLSettings : public QWidget, private Ui::Settings
 {
     Q_OBJECT

 public:
	 GLSettings(QWidget *parent = 0);

 private slots:
     void onClearInBlackButtonClicked();
     void onWireframeButtonClicked();

 private:
	 void callUpdateCB();

	 bool m_ClearInBlackConfig;
	 bool m_WireframeConfig;

	 UpdateCB m_UpdateCB;
	 void * m_UpdatePtr;

 public: // Accessors
	 bool getClearInBlackConfig() const
	 {
		 return m_ClearInBlackConfig;
	 }
	 bool getWireframeConfig() const
	 {
		 return m_WireframeConfig;
	 }
	 void setUpdateCB( UpdateCB a_UpdateCB, void * a_UpdatePtr )
	 {
		 m_UpdateCB = a_UpdateCB;
		 m_UpdatePtr = a_UpdatePtr;
	 }
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
