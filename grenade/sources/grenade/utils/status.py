#!/usr/bin/python2.5
#                 Dr. D Studios - Software Disclaimer
#
# Copyright 2009 Dr D Studios Pty Limited (ACN 127 184 954) (Dr. D Studios), its
# affiliates and/or its licensors.
#

def get_pipeline_step_task_status_summary(status_list):
    """
    
    There is a new branch of the Python API in GitHub which supports a new 
    feature or method for 'summaries'.  This should make it possible to get 
    task summaries (similar to http://shotgun/page/8172) without a lot of client
    side processing.   And indeed, it works a charm... except for one small 
    detail...
    
    While (for example) it does return the aggregated list of tasks status for 
    a shot and pipeline step, it doesn't take this one step further and reduce 
    it to a single task summary as in the UI.  So, for the the Stereo pipeline
    step on shot 01a_010 in Happy Feet 2, the new 'summarize' method will gladly
    return the following::
    
        >>> {'sg_status_list': 'conapr,dlvr,omt,wtg'}
    
    but not what we actually want which is just a simple 'ip'.  And here is the 
    kicker... they can't actually do that summary for you because... wait for 
    it... that logic is client specific and is encapsulated in a nice piece of
    javascript code - base_28538.js to be precise, you'll find it around line 
    2640 or thereabouts.  If you can't find it there, just search the javascript
    code for 'custom_for=="drd"' (the same applies to the ruby logic also).  For
    the record, and at the time of writing (21/09/2011 20:39:28), this logic 
    looks something like
    
    .. code-block:: javascript
        :linenos:
        
        if(ShotgunGlobals.custom_for=="drd"){
            if(vals.length==0){
                status='na';
            }else if(vals.length==1){
                status=vals[0];
            }else{
                var vals=vals.filter(function ignoreit(e){return(['omt','clsd'].indexOf(e)==-1)});
                if(vals.length==1){
                    status=vals[0];
                }else{
                    if(vals.every(function checkit(ele){return this.indexOf(ele)>-1},['inv','rdy','wtg'])){
                        status='inv';
                    }else if(vals.every(function checkit(ele){return this.indexOf(ele)>-1},['wtg','hld'])){
                        status='hld';
                    }else if(vals.every(function checkit(ele){return this.indexOf(ele)>-1},['conapr','dap'])){
                        status='conapr';
                    }else{
                        status='ip';
                    }
                }
            }
            
            if(vals.length==1){
                if(vals[0]=='rdy'){
                    status='inv';
                }
            }
        }
    
    Note however, that I have unobfuscated this code for you - depending on when
    you try and repeat this process you will have to unminify and uncompress the
    piece of code in question based on whatever minifier and compressor they are
    using at the time.  It's fun - really, you might just have to trust me on 
    that one.
    
    As a result, if we ask Shotgun Software to change the summary logic based on 
    some ad. hoc. non-sensical logic required by production, this code breaks...
    busted... kaput.  We have to go and repeat this whole process, all, over, 
    again.  But please don't ask me to do it for you - I will laugh.
    
    You'll notice too that the output of the 'summarize' method is a nice comma 
    separated string.  I mean, come on, how hard is it to return that as a 
    proper list - just for simplicities sake. 
    
    So, what does this function actually do.  Well, it replicates that little 
    bit of javascript in Python.  You can feed it the comma separated list 
    (sorry, I mean string) output from the 'summarize' method and it will reduce
    it to a single status value based on whatever crack Shotgun Software are 
    using today.
    
    I even had to add my own comments.
    
    .. versionadded:: 1.7.0
    .. versionchanged:: 1.7.1
        Shotgun Software changed their logic, so of course, must I.
    
    :param status_list:
        A list of status values, for example, ['conapr','dlvr','omt','wtg'].
    :returns:
        A single status value as a string.
    """
    
    if not status_list:
        return 'na'
    
    # Strip out any status values which we don't care about.
    status_list = [item for item in status_list if item not in ['na', '', None]]
    
    # If that means we have no values left, then 'na' is our default.
    if len(status_list) == 0 :
        status = 'na'
    
    # Or, if it means we have just one value left, then we use this.
    elif len(status_list) == 1:
        status = status_list[0]
    
    # Or, if we have more than one, we do some further filtering...
    else:
        # Some more status values we don't care about and so filter away...
        status_list = [item for item in status_list if item not in ['omt', 'clsd']]
        
        # If that means we have just one value left, then we use this.
        if len(status_list) == 1:
            status = status_list[0]
        
        # Or perform some logic to come up with a different value.
        else:
            if not set(status_list).difference(set(['inv', 'rdy', 'wtg'])):
                status = 'inv'
            elif not set(status_list).difference(set(['wtg', 'hld'])):
                status = 'hld'
            elif not set(status_list).difference(set(['conapr', 'dap'])):
                status = 'conapr'
            else:
                status = 'ip'
    
    # Now, I'll be honest, I can't work out why this step is necessary.  Why, if
    # we only have one value, would we want to fake what that value is - why 
    # as a production person with all your worldly experience, would you want a
    # single task with task status 'rdy' to appear as 'inv'?  Answers on a 
    # postcard please.
    if len(status_list) == 1:
        if status_list[0] == 'rdy':
            status = 'inv'
    
    # And finally, return the status value we wanted originally.  Thank you, and
    # goodnight.
    return status

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

