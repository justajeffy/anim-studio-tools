##
#   \namespace  reviewTool.kernel
#
#   \remarks    Creates a central QObject signal dispatcher for emitting signals throughout the
#               Review Tool application that need to be non-specific as to the source
#   
#   \author     Dr. D Studios
#   \date       08/08/11
#

from PyQt4.QtCore   import  pyqtSignal,\
                            QObject

class Kernel(QObject):
    errorReported   = pyqtSignal(str)
    infoReported    = pyqtSignal(str)
    warningReported = pyqtSignal(str)
    debugReported   = pyqtSignal(str)
    progressUpdated = pyqtSignal(int)
    
    def debug( self, text ):
        """
                Emits the warning reported signal
                to any object that might be listening,
                provided this object is not blocking signals.
                
                :param      text:
                :type       <str>:
        """
        if ( not self.signalsBlocked() ):
            self.debugReported.emit('[DEBUG]: %s\n' % text)
    
    def info( self, text ):
        """
                Emits the info reported signal
                to any object that might be listening,
                provided this object is not blocking signals.
                
                :param      text:
                :type       <str>:
        """
        if ( not self.signalsBlocked() ):
            self.infoReported.emit('%s\n' % text)
    
    def warn( self, text ):
        """
                Emits the warning reported signal
                to any object that might be listening,
                provided this object is not blocking signals.
                
                :param      text:
                :type       <str>:
        """
        if ( not self.signalsBlocked() ):
            self.warningReported.emit('[WARNING]: %s\n' % text)
    
    def updateProgress( self, percent ):
        """
                Emits the progress updated signal
                to any object that might be listening,
                provided this object is not blocking signals.
                
                :param      text:
                :type       <str>:
        """
        if ( not self.signalsBlocked() ):
            self.progressUpdated.emit(percent)

# create the kernel instance
core = Kernel()
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

