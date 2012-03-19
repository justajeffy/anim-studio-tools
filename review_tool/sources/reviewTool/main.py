##
#   \namespace  reviewTool.main
#
#   \remarks    The main executable method for running the Review Tool utility as a standalone
#               application
#   
#   \author     Dr. D Studios
#   \date       08/04/11
#

import os

from optparse               import OptionParser

from PyQt4.QtGui            import QApplication
from drGadgets.gui.DrSplash import DrSplash

from reviewTool            import resources

def command(argv):
    """
    Executes command line utility for the review tool
    """
    # parse out the general options
    usage = []
    usage.append( 'reviewtool [options] clip1 clip2 ...' )
    usage.append( '' )
    usage.append( 'Clips are space separated arguments and can be formatted as:' )
    usage.append( '' )
    usage.append( 'CLIP_ID:VIDEO_SOURCE:AUDIO_SOURCE - VIDEO_SOURCE and AUDIO_SOURCE are optional' )
    usage.append( '' )
    usage.append( 'The CLIP_ID can be broken down as:' )
    usage.append( 'SHOT                     (e.g.: reviewtool -d light 01a_010 01a_020' )
    usage.append( 'SHOT-DEPT                (e.g.: reviewtool 01a_010-lens 01a_020-light' )
    usage.append( 'SHOT-DEPT[VERSION]       (e.g.: reviewtool 400_010-lens[058], defaults to latest)' )
    usage.append( 'SHOT-DEPT[VERSION:LIMIT] (e.g.: reviewtool 400_010-lens[:1], latest & 1 before latest, order goes [latest,latest-1,..,earliest])' )
    
    parser = OptionParser(usage='\n'.join(usage))
    
    # add the comandline options
    parser.add_option('',       '--ui',         dest = 'gui',           action='store_true',    help = 'launch the gui instead of automatically playing', default = False )
    parser.add_option('-p',     '--playlist',   dest = 'playlist',                              help = 'loads a playlist file for review', default = '' )
    parser.add_option('-d',     '--dept',       dest = 'dept',                                  help = 'can be used to specify override the department-per-clip (e.g: lens or lens,light)', default = '' )
    parser.add_option('-m',     '--mode',       dest = 'mode',                                  help = 'defines the play mode (movie|image)' )
    parser.add_option('-a',     '--audio',      dest = 'audio',         action='store_true',    help = 'set the clip to play with audio overrides (0|1)',   default = False )
    parser.add_option('',       '--pad',        dest = 'pad',                                   help = 'pad the inputed clips with the number of siblings provided by the pad amount' )
    parser.add_option('',       '--padLeft',    dest = 'padLeft',                               help = 'pad the inputed clips with the number of siblings found before before the current clip' )
    parser.add_option('',       '--padRight',   dest = 'padRight',                              help = 'pad the inputed clips with the number of siblings found after the current clip' )
    parser.add_option('-c',     '--compare',    dest = 'compare',                               help = 'defines the compare mode when loading multiple shots (layout|layout_packed|stack_blend|stack_wipe)' )
    
    # use command line options for config overrides
    (options, clips) = parser.parse_args(argv)
    
    # create the padding information
    if ( options.padLeft == None ):
        options.padLeft = options.pad if options.pad != None else 0
    if ( options.padRight == None ):
        options.padRight = options.pad if options.pad != None else 0

    options.padLeft     = int(options.padLeft)
    options.padRight    = int(options.padRight)
    
    # determin if the gui should be loaded up and the command line was just
    # suppling additional arguments
    if ( options.gui ):
        return (options,clips)
    
    from reviewTool.api.entity import Entity
    session = Entity.quickLaunch( clips,
                        playlist            = options.playlist,
                        defaultDepartment   = options.dept,
                        padLeft             = options.padLeft,
                        padRight            = options.padRight,
                        overrideAudio       = options.audio,
                        compareMethod       = options.compare,
                        mode                = options.mode 
                        )
    
    # disconnect from the session
    if ( session ):
        session._rvc.disconnect()
    
    return (options,clips)

def main(argv):
    """
    Executes the main event loop for the method
    
    :return:    <int>   error code
    """ 
    # process command line mode
    if ( len(argv) > 1 ):
        options,clips = command(argv[1:])
        
        # don't load the interface based on the command line args
        if ( not options.gui ):
            return 0
    else:
        options,clips = (None,None)
        
    # create the QApplication
    app     = QApplication(argv)
    app.setStyle('plastique')
    
    splash  = DrSplash(title="Review Tool",
                      version="Ver "+ os.environ.get("REVIEW_TOOL_VER", ""),
                      icon_path = resources.find('img/icon.png')
                      )
    
    # set the application's icon
    app.setWindowIcon( resources.icon('img/icon.png'))
    
    splash.log("loading Tank...")
    import tank
    
    splash.log("initializing gui...")
    
    from reviewTool.gui.reviewtoolwindow import ReviewToolWindow
    window = ReviewToolWindow()
    window.show()
    window.setWindowTitle( 'Review Tool - %s [%s]' % (os.environ.get('REVIEW_TOOL_VER',''),os.environ.get('DRD_JOB','').upper()) )
    splash.finish(window)
    
    # launch command arguments
    window.initializeFromCLI( options, clips )
    
    # run the application
    return app.exec_()
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

