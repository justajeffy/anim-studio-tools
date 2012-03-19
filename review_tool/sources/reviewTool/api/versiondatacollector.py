##
#   \namespace  reviewTool.api.versiondatacollector
#
#   \remarks    [desc::commented]
#   
#   \author     Dr. D Studios
#   \date       08/05/11
#

import os
import tank
from rv_tools.util  import get_frame_sequence_source

from ..kernel       import core

#--------------------------------------------------------------------------------

def collectAudioData(address):
    try:
        tank_audio  = tank.find(address)
    except:
        tank_audio = None
        core.warn( 'Could not find tank address: %s...' % address )
    
    audioSource = ''
    audioStart  = 0
    audioOffset = 0
    
    # load data from the tank audio
    if ( tank_audio ):
        pipeline_data       = tank_audio.properties.get('pipeline_data',{})
        
        audioSource         = tank_audio.system.filesystem_location
        audioStart          = pipeline_data.get('start_frame',-1)
        audioOffset         = pipeline_data.get('start_offset',0)
    
    return (audioSource,audioStart,audioOffset)

def collectFrameData(address):
    frames_address  = address.replace('Movie(','Frames(')
    try:
        tank_frames = tank.find(frames_address)
    except:
        tank_frames = None
        core.warn( 'Could not find tank address: %s...' % address )
    
    imageSource = ''
    sourceStart = 0
    sourceEnd   = 0
    stereo_pair = []
    
    if ( tank_frames ):
        filepath                    = tank_frames.system.filesystem_location
        
        # extract the stereo pair information from the tank frames object
        stereo_pair                 = tank_frames.properties['stereo_pair']
        if ( not stereo_pair ):
            stereo_pair = ['left']
            
        spath, sname, smin, smax    = get_frame_sequence_source(filepath)
        
        if ( not (spath and sname) ):
            core.warn( 'Could not extract frame data from %s' % filepath )
        else:
            imageSource = os.path.join(spath,sname)
            sourceStart = smin
            sourceEnd   = smax
    
    return (imageSource,sourceStart,sourceEnd,stereo_pair)

def collectMovieData(address):
    movie_address  = address.replace('Frames(','Movie(')
    try:
        tank_movie = tank.find(movie_address)
    except:
        tank_movie = None
        core.warn( 'Could not find tank address: %s...' % address )
    
    # load data from the tank movie
    videoSource = ''
    if ( tank_movie ):
        videoSource = tank_movie.system.filesystem_location
        
    return videoSource

#--------------------------------------------------------------------------------

if ( __name__ == '__main__' ):
    import sys

    address         = sys.argv[0]
    audio_address   = sys.argv[1]
    
    imageSource, sourceStart, sourceEnd, stereo_pair    = collectFrameData(address)
    videoSource                                         = collectMovieData(address)
    audioSource, audioStart, audioOffset                = collectAudioData(audio_address)
    
    print 'imageSource:%s'  % imageSource
    print 'sourceStart:%s'  % sourceStart
    print 'sourceEnd:%s'    % sourceEnd
    print 'stereo_pair:%s'  % stereo_pair
    print 'videoSource:%s'  % videoSource
    print 'audioSource:%s'  % audioSource
    print 'audioStart:%s'   % audioStart
    print 'audioOffset:%s'  % audioOffset

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

