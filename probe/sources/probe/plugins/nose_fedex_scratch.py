#        Dr. D Studios - Software Disclaimer
#
# Copyright 2010 Dr D Studios Pty Limited (ACN 127 184 954) (Dr. D Studios), its
# affiliates and/or its licensors.
#

import itch.fedex_scratch
import nose.plugins

class FedexScratchNosePlugin(nose.plugins.Plugin):
    """
    Adds Fedex scratch setup and teardown capability to tests.
    
    .. versionadded:: 0.11.0
    """
    
    name = 'fedex-scratch'
    score = 1 # run early
    
    def __init__(self):
        """
        .. versionadded:: 0.11.0
        """
        
        super(FedexScratchNosePlugin, self).__init__()
        
        self.preserve = False
        self.scratch_name = None
    
    def options(self, parser, env):
        """
        Register commandline options.
        
        .. versionadded:: 0.11.0
        """
        
        super(FedexScratchNosePlugin, self).options(parser, env)
        
        parser.add_option('--fedex-scratch-preserve', action='store_true', dest='preserve',
                          default=False,
                          metavar="PRESERVE",
                          help="Only create new scratch if missing and don't teardown at end")
        
        parser.add_option('--fedex-scratch-name', action='store', dest='scratch_name',
                          default=None,
                          metavar="NAME",
                          help="A name for the scratch area.")
    
    def available(self):
        """
        Check to see if this plugin is available.
        
        .. versionadded:: 0.12.0
        """
        
        fs = itch.fedex_scratch.FedexScratch()
        fs.available()
    
    def configure(self, options, conf):
        """
        Configure plugin.
        
        .. versionadded:: 0.11.0
        """
        
        super(FedexScratchNosePlugin, self).configure(options, conf)
        
        self.conf = conf
        self.preserve = options.preserve
        self.scratch_name = options.scratch_name
        
        self.fs = itch.fedex_scratch.FedexScratch(name=self.scratch_name)
    
    def begin(self):
        """
        Setup scratch area if required.
        
        .. versionadded:: 0.11.0
        """
        
        if self.preserve:
            if not self.fs.exists():
                self.fs.setup()
        
        else:
            self.fs.setup(auto_teardown=True)
    
    def finalize(self, result):
        """
        Tear down scratch area if required.
        
        .. versionadded:: 0.11.0
        """
        
        if not self.preserve:
            self.fs.teardown()

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

