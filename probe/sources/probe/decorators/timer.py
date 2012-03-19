# 
# Copyright 2009 Dr D Studios Pty Limited (ACN 127 184 954) (Dr. D Studios), its
# affiliates and/or its licensors.
# 

import functools
import os
import time

def timer(iterations=1, output_directory="/tmp"):
    def func_(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            # Build the path to the folder where we want to keep the results
            # Make sure this directory exists
            if not os.path.isdir(output_directory):
                os.makedirs(output_directory)
            
            csv_file = os.path.join(
                                    output_directory, 
                                    "%s_timer_results.csv" % (f.__name__)
                                   )
            
            fd = open(csv_file, "w")
            
            # Call the wrapped function for the correct number of iterations.
            for i in range(iterations):
                start = time.time()
                
                _ = f(*args, **kwargs)
                
                end = time.time()
                elapsed = (end - start) * 1000.0
                
                fd.write("%s,%s\n" % (i, elapsed))
            
            fd.close()
        
        return wrapper
    
    return func_

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

