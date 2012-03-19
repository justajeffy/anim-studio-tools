#        Dr. D Studios - Software Disclaimer
#
# Copyright 2010 Dr D Studios Pty Limited (ACN 127 184 954) (Dr. D Studios), its
# affiliates and/or its licensors.
#

import itch.tank_scratch
import nose.plugins

class TankScratchNosePlugin(nose.plugins.Plugin):
    """
    Adds Tank scratch setup and teardown capability to tests.
    
    .. versionadded:: 0.6.0
    """
    
    name = 'tank-scratch'
    score = 1 # run early
    
    def __init__(self):
        """
        .. versionadded:: 0.6.0
        """
        
        super(TankScratchNosePlugin, self).__init__()
        
        self.preserve = False
        self.scratch_name = None
    
    def options(self, parser, env):
        """
        Register commandline options.
        
        .. versionadded:: 0.6.0
        """
        
        super(TankScratchNosePlugin, self).options(parser, env)
        
        parser.add_option('--tank-scratch-preserve', action='store_true', dest='preserve',
                          default=False,
                          metavar="PRESERVE",
                          help="Only create new scratch if missing and don't teardown at end")
        
        parser.add_option('--tank-scratch-name', action='store', dest='scratch_name',
                          default=None,
                          metavar="NAME",
                          help="A name for the scratch area.")
    
    def available(self):
        """
        Check to see if this plugin is available.
        
        .. versionadded:: 0.12.0
        """
        
        ts = itch.tank_scratch.TankScratch()
        ts.available()
    
    def configure(self, options, conf):
        """
        Configure plugin.
        
        .. versionadded:: 0.6.0
        .. versionchanged:: 0.9.0
            Changed the module name.
        """
        
        super(TankScratchNosePlugin, self).configure(options, conf)
        
        self.conf = conf
        self.preserve = options.preserve
        self.scratch_name = options.scratch_name
        
        self.ts = itch.tank_scratch.TankScratch(name=self.scratch_name)
    
    def begin(self):
        """
        Setup scratch area if required.
        
        .. versionadded:: 0.6.0
        .. versionchanged:: 0.9.1
            Updating project name to match what is used elsewhere.
        """
        
        if self.preserve:
            if not self.ts.exists():
                self.ts.setup()
        
        else:
            self.ts.setup(auto_teardown=True)
        
        os.environ["TANK_CONFIG_TANK_SCRATCH"] = os.path.join(self.ts.location, "scratch_config.cfg")
        os.environ["TANK_PROJECT"] = "TANK_SCRATCH"
    
    def finalize(self, result):
        """
        Tear down scratch area if required.
        
        .. versionadded:: 0.6.0
        """
        
        if not self.preserve:
            self.ts.teardown()

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

