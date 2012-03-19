
#-----------------------------------------------------------------------------
import datetime
import os
import types
from PyQt4 import QtGui, QtCore, uic
from rodin import logging
log = logging.get_logger('grind.concorde.ui')

#-----------------------------------------------------------------------------
def ui_load( name ):
    template = uic.loadUiType(os.path.join(os.path.split(__file__)[0], "ui", name ))[0]
    return template

#-----------------------------------------------------------------------------
class ConcordeWindow(ui_load("Concorde.ui"), QtGui.QMainWindow):
    def __init__(self):
        QtGui.QMainWindow.__init__(self)
        self.setupUi(self)

#-----------------------------------------------------------------------------
class AttribBase:

#-----------------------------------------------------------------------------
    def setRanges(self,value):
        if value<self.valueSB.minimum():
            self.valueSB.setMinimum(value)
        if value>self.valueSB.maximum():
            self.valueSB.setMaximum(value)

        if value<self.valueSlider.minimum():
            self.valueSlider.setMinimum(value)
        if value>self.valueSlider.maximum():
            self.valueSlider.setMaximum(value)

        if value<self.rangeMin.value():
            self.rangeMin.setValue(value)
        if value>self.rangeMax.value():
            self.rangeMax.setValue(value)

#-----------------------------------------------------------------------------
    def onValueChangedSB(self):
        value=self.valueSB.value()
        self.valueSlider.setValue(value)
        self.setRanges(value)
        self.onValueChanged(value)

#-----------------------------------------------------------------------------
    def onValueChangedSlider(self):
        value=self.valueSlider.value()
        self.valueSB.setValue(value)
        self.setRanges(value)
        self.onValueChanged(value)

#-----------------------------------------------------------------------------
    def onRangeMinChanged(self,value):
        self.valueSB.setMinimum(value)
        self.valueSlider.setMinimum(value)

#-----------------------------------------------------------------------------
    def onRangeMaxChanged(self,value):
        self.valueSB.setMaximum(value)
        self.valueSlider.setMaximum(value)

#-----------------------------------------------------------------------------
    def onValueChanged(self,value):
        if self.callback is None:
            return
        if type(self.callback) == types.ListType:
            for cb in self.callback:
                cb(value)
        else:
            self.callback(value)

#-----------------------------------------------------------------------------
class AttribIntWidget(ui_load( "AttribIntWidget.ui" ),QtGui.QWidget,AttribBase):
    def __init__(self,parent,name,value,extra,callback=None, canUseSlider=True):
        QtGui.QWidget.__init__(self,parent)
        self.setupUi(self)
        self.label.setText(name+":")

        self.setRanges(value)
        self.valueSlider.setValue(value)
        self.valueSB.setValue(value)
        self.ranges.setVisible(False)

        self.connect(self.rangeMin, QtCore.SIGNAL('valueChanged(int)'), self.onRangeMinChanged)
        self.connect(self.rangeMax, QtCore.SIGNAL('valueChanged(int)'), self.onRangeMaxChanged)
        self.connect(self.valueSB, QtCore.SIGNAL('valueChanged(int)'), self.onValueChangedSB)
        if canUseSlider == True:
            self.connect(self.valueSlider, QtCore.SIGNAL('valueChanged(int)'), self.onValueChangedSlider)
        else:
            self.valueSlider.setEnabled( False )

        self.callback=callback
        if extra is not None:
            self.set_min_max(extra["min"],extra["max"],extra["min_hard"],extra["max_hard"])

#-----------------------------------------------------------------------------
    def set_value(self,value):
        self.valueSB.setValue(value)

    def get_value(self):
        return self.valueSB.value()

#-----------------------------------------------------------------------------
    def set_min_max(self,min,max,min_hard,max_hard):
        self.min=min
        self.max=max
        self.min_hard=min_hard
        self.max_hard=max_hard

        self.valueSB.setMinimum(self.min)
        self.valueSB.setMaximum(self.max)

        self.valueSlider.setMinimum(self.min)
        self.valueSlider.setMaximum(self.max)

        self.rangeMin.setValue(self.min)
        self.rangeMin.setMinimum(self.min_hard)
        self.rangeMin.setMaximum(self.max_hard)

        self.rangeMax.setValue(self.max)
        self.rangeMax.setMinimum(self.min_hard)
        self.rangeMax.setMaximum(self.max_hard)

#-----------------------------------------------------------------------------
class AttribFloatWidget(ui_load( "AttribFloatWidget.ui" ),QtGui.QWidget):
    def __init__(self,parent,name,value,extra,callback=None):
        QtGui.QWidget.__init__(self,parent)
        self.setupUi(self)
        self.label.setText(name+":")
        self.veto = False

        # Since Qt is lame and doesn't have a QDoubleSlider where we can set the values nicely, we have to
        # do this abomination

        self.ranges.setVisible(False)

        self.connect(self.rangeMin, QtCore.SIGNAL('valueChanged(double)'), self.onRangeMinChanged)
        self.connect(self.rangeMax, QtCore.SIGNAL('valueChanged(double)'), self.onRangeMaxChanged)
        self.connect(self.value, QtCore.SIGNAL('editingFinished()'), self.onValueTextChanged)
        self.connect(self.valueSlider, QtCore.SIGNAL('valueChanged(int)'), self.onValueSliderChanged)
        self.callback=callback

        self.range = 10.0
        self.singleStep = self.range / 10000.0

        if extra is not None:
            self.set_min_max(extra["min"],extra["max"],extra["min_hard"],extra["max_hard"])

        self.setValue(value)

#-----------------------------------------------------------------------------
    def setValue(self,value):
        self.veto = True
        try:
            self.valueSlider.setValue(self.f2i(value))
            self.value.setText(str(value))
        finally:
            self.veto = False

#-----------------------------------------------------------------------------
    def set_min_max(self,min,max,min_hard,max_hard):
        self.veto = True
        try:
            self.range = max - min

            self.min=min
            self.max=max
            self.min_hard=min_hard
            self.max_hard=max_hard

            self.valueSlider.setMinimum(self.f2i(self.min))
            self.valueSlider.setMaximum(self.f2i(self.max))

            self.rangeMin.setValue(self.min)
            self.rangeMin.setMinimum(self.min_hard)
            self.rangeMin.setMaximum(self.max_hard)

            self.rangeMax.setValue(self.max)
            self.rangeMax.setMinimum(self.min_hard)
            self.rangeMax.setMaximum(self.max_hard)
        finally:
            self.veto = False

#-----------------------------------------------------------------------------
    def i2f(self,i):
        return i * self.singleStep

#-----------------------------------------------------------------------------
    def f2i(self,f):
        return f / self.singleStep

#-----------------------------------------------------------------------------
    def setRanges(self,value):
        valueI = self.f2i(value)
        if valueI<self.valueSlider.minimum():
            self.valueSlider.setMinimum(valueI)
        if valueI>self.valueSlider.maximum():
            self.valueSlider.setMaximum(valueI)

        if value<self.rangeMin.value():
            self.rangeMin.setValue(value)
        if value>self.rangeMax.value():
            self.rangeMax.setValue(value)

#-----------------------------------------------------------------------------
    def revertValue(self):
        value=self.i2f(self.valueSlider.value())
        self.value.setText(str(value))

#-----------------------------------------------------------------------------
    def onValueTextChanged(self):
        if self.veto: return
        t = self.value.text()
        value = 0.0
        try:
            value = float(t)
        except ValueError:
            self.revertValue()
            return

        if value < self.min or value > self.max:
            log.error("value out of bounds: %f [%f,%f]"%(value,self.min,self.max))
            self.revertValue()
            return

        self.valueSlider.setValue(self.f2i(value))
        self.onValueChanged(value)

#-----------------------------------------------------------------------------
    def onValueSliderChanged(self):
        if self.veto: return
        value=self.i2f(self.valueSlider.value())
        self.value.setText(str(value))
        self.onValueChanged(value)

#-----------------------------------------------------------------------------
    def onRangeMinChanged(self,value):
        if self.veto: return
        self.min = self.f2i(value)
        self.valueSlider.setMinimum(self.min)

#-----------------------------------------------------------------------------
    def onRangeMaxChanged(self,value):
        if self.veto: return
        self.max = self.f2i(value)
        self.valueSlider.setMaximum(self.max)


#-----------------------------------------------------------------------------
    def onValueChanged(self,value):
        if self.veto: return
        if self.callback is None:
            return
        if type(self.callback) == types.ListType:
            for cb in self.callback:
                cb(value)
        else:
            self.callback(value)


#-----------------------------------------------------------------------------
class FrameWidget(ui_load( "FrameWidget.ui" ),QtGui.QWidget,AttribBase):
    def __init__(self,parent,name,value,extra,callback=None):
        QtGui.QWidget.__init__(self,parent)
        self.setupUi(self)
        self.label.setText(name+":")
        self.set_value(value)
        self.callback=callback
        if extra is not None: self.set_min_max(extra["min"],extra["max"],extra["min_hard"],extra["max_hard"])
        self.connect(self.value, QtCore.SIGNAL('editingFinished()'), self.onValueTextChanged)
        self.connect(self.prevButton, QtCore.SIGNAL('clicked()'), self.onPrevButtonClicked)
        self.connect(self.nextButton, QtCore.SIGNAL('clicked()'), self.onNextButtonClicked)

    def set_value(self,value):
        self.value.setText(str(value))

    def onValueTextChanged(self):
        if self.callback is None: return
        try:
            self.value.setEnabled( False )
            self.callback( int(self.value.text()), False )
            self.value.setEnabled( True )
        except:
            self.value.setEnabled( True )
            return

    def onPrevButtonClicked(self):
        if self.callback is None: return
        try:
            self.callback( int(self.value.text())-1, True )
        except ValueError:
            return

    def onNextButtonClicked(self):
        if self.callback is None: return
        try:
            self.callback( int(self.value.text())+1, True )
        except ValueError:
            return

    def set_min_max(self,min,max,min_hard,max_hard):
        self.min=min
        self.max=max

# Copyright 2008-2012 Dr D Studios Pty Limited (ACN 127 184 954) (Dr. D Studios)
#
# This file is part of anim-studio-tools.
#
# anim-studio-tools is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# anim-studio-tools is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with anim-studio-tools.  If not, see <http://www.gnu.org/licenses/>.

