# 
# Copyright 2009 Dr D Studios Pty Limited (ACN 127 184 954) (Dr. D Studios), its
# affiliates and/or its licensors.
# 

import cProfile
import functools
import gprof2dot
import os
import pydot
import sys

def profile(graph=False, output_directory="/tmp"):
    def func_(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            # Build the path to the folder where we want to keep the results
            # Make sure this directory exists
            if not os.path.isdir(output_directory):
                os.makedirs(output_directory)
            
            # Run and profile the function, dumping the resulting stats into a
            # file in the results directory.
            pstats_file = os.path.join(
                                       output_directory, 
                                       "%s_pstats_results.pstats" % (f.__name__)
                                      )
            
            profiler = cProfile.Profile()
            profiler.runcall(f, *args, **kwargs)
            profiler.dump_stats(pstats_file)
            
            # Dump the profile stats in a human readable for to a file on 
            # disk
            stats_file = os.path.join(
                                      output_directory, 
                                      "%s_profiler_results.stats" % (f.__name__)
                                     )
            
            # Temporarily redirect stdout to somewhere else
            fd = open(stats_file, "w")
            
            _stdout = sys.stdout
            sys.stdout = fd
            
            # Dump the stats to this file
            profiler.print_stats()
            
            # Point stdout back to what it should be
            sys.stdout = _stdout
            fd.close()
            
            # Now we want to dump the stats to a pretty graph.
            if graph:
                class FakeOptions(object):
                    """A fake options object to trick the gprof2dot main method."""
                    pass
                
                # Setup an instance of gprof2dot, forcing some options that 
                # would usually come through the command line.
                gprof = gprof2dot.Main()
                
                gprof.options = FakeOptions()
                gprof.options.node_thres = 0.5
                gprof.options.edge_thres = 0.1
                gprof.options.wrap = False
                gprof.options.strip = False
                gprof.options.theme = "color"
                
                gprof.theme = gprof2dot.Main.themes[gprof.options.theme]
                
                # Now build the graph
                parser = gprof2dot.PstatsParser(pstats_file)
                gprof.profile = parser.parse()
                
                dot_file = os.path.join(
                                        output_directory, 
                                        "%s_dot_results.dot" % (f.__name__)
                                       )
                
                gprof.output = open(dot_file, 'wt')
                gprof.write_graph()
                gprof.output.close() 
                
                dot_graph = pydot.graph_from_dot_file(dot_file)
                
                png_file = os.path.join(
                                        output_directory, 
                                        "%s_profiler_graph.png" % (f.__name__)
                                       )
                
                dot_graph.write(png_file, format="png", prog="dot")
        
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

