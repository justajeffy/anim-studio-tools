##
#   \namespace  reviewTool.util.diagnosis
#
#   \remarks    Creates diagnosis information about data
#   
#   \author     eric.hulser@drdstudios.com
#   \author     Dr. D Studios
#   \date       08/16/11
#

import ConfigParser
import os
import subprocess
import sys
import time

from reviewTool import settings

from PyQt4.QtCore import QThread

class DiagnosisThread(QThread):
    def __init__( self, collection ):
        super(DiagnosisThread,self).__init__()
        self.collection = collection
        
    def run( self ):
        # save the inputed collection to disk
        parser = ConfigParser.ConfigParser()

        for key, entries in self.collection.items():
            parser.add_section(key)
            for i, entry in enumerate(entries):
                option = 'entry%i' % i
                parser.set(key,option, '%s|%s' % entry)

        # save the data to file
        tempFile = settings.tempPath( 'comparison_config.ini' )
        f = open( tempFile, 'w' )
        parser.write( f )
        f.close()

        # run the comparison in a separate process
        proc = subprocess.Popen( 'python2.5 %s %s' % (__file__,tempFile), shell = True )
        proc.communicate()
#        generateReport(filename = tempFile)

#--------------------------------------------------------------------------------

def generateReport( filename = '', collection = {} ):
    import bkdDiagn.sg_bkdDiagn
    
    # load the data from file
    if ( filename ):
        parser = ConfigParser.ConfigParser()
        parser.read(filename)
        
        # create the collection
        collection = {}
        for section in parser.sections():
            data = []
            for option in parser.options(section):
                value = parser.get(section,option)
                result = value.split('|')
                if ( len(result) != 2 ):
                    continue
                data.append( result )
            collection[section] = data
    
    if ( not collection ):
        return False
    
    # process the results
    result          = bkdDiagn.sg_bkdDiagn.batch_process2(collection.items())
    html_file_name  = time.ctime().replace( ' ', '_' ).replace( ':', '-' ) + '.html'
    html_file_path  = settings.desktopPath( 'breakdown_diagnosis' )
    if ( not os.path.exists( html_file_path ) ):
        os.mkdir( html_file_path )
    
    filename = os.path.join( html_file_path, html_file_name )
    
    # save the html data to file
    f = open( filename, 'w' )
    f.write( result )
    f.close()
    
    return True
    
if ( __name__ == '__main__' ):
    generateReport( filename = sys.argv[1] )
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

