/********************************************************************************
** Form generated from reading ui file 'settings.ui'
**
** Created: Mon Oct 26 16:34:15 2009
**      by: Qt User Interface Compiler version 4.5.3
**
** WARNING! All changes made in this file will be lost when recompiling ui file!
********************************************************************************/

#ifndef UI_SETTINGS_H
#define UI_SETTINGS_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QCheckBox>
#include <QtGui/QHeaderView>
#include <QtGui/QPushButton>
#include <QtGui/QWidget>

QT_BEGIN_NAMESPACE

class Ui_Settings
{
public:
    QPushButton *assignRandomClearColor;
    QCheckBox *ui_Wireframe;
    QCheckBox *clearInBlack;

    void setupUi(QWidget *Settings)
    {
        if (Settings->objectName().isEmpty())
            Settings->setObjectName(QString::fromUtf8("Settings"));
        Settings->resize(405, 273);
        assignRandomClearColor = new QPushButton(Settings);
        assignRandomClearColor->setObjectName(QString::fromUtf8("assignRandomClearColor"));
        assignRandomClearColor->setGeometry(QRect(20, 40, 171, 27));
        ui_Wireframe = new QCheckBox(Settings);
        ui_Wireframe->setObjectName(QString::fromUtf8("ui_Wireframe"));
        ui_Wireframe->setGeometry(QRect(20, 80, 121, 22));
        clearInBlack = new QCheckBox(Settings);
        clearInBlack->setObjectName(QString::fromUtf8("clearInBlack"));
        clearInBlack->setGeometry(QRect(20, 110, 121, 22));

        retranslateUi(Settings);

        QMetaObject::connectSlotsByName(Settings);
    } // setupUi

    void retranslateUi(QWidget *Settings)
    {
        Settings->setWindowTitle(QApplication::translate("Settings", "Form", 0, QApplication::UnicodeUTF8));
        assignRandomClearColor->setText(QApplication::translate("Settings", "AssignRandomClearColor", 0, QApplication::UnicodeUTF8));
        ui_Wireframe->setText(QApplication::translate("Settings", "Wireframe", 0, QApplication::UnicodeUTF8));
        clearInBlack->setText(QApplication::translate("Settings", "Clear In Black", 0, QApplication::UnicodeUTF8));
        Q_UNUSED(Settings);
    } // retranslateUi

};

namespace Ui {
    class Settings: public Ui_Settings {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_SETTINGS_H


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
