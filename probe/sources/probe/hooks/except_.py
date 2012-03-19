# 
# Copyright 2009 Dr D Studios Pty Limited (ACN 127 184 954) (Dr. D Studios), its
# affiliates and/or its licensors.
# 

"""
Add a hook to call the pdb debugger in post mortem mode on uncaught exceptions.

To enable this in your application:

    >>> import probe.hooks.except_hook
    >>> probe.hooks.except_hook.init()

.. versionadded:: 0.1.0

.. note::
    This has been taken from :extref:`google-apputils<GoogleAppUtils>` and 
    adapted slightly to fit the Dr. D. Studios environment.  This is licensed 
    under :extref:`Apache 2.0<Apache2>`.
"""

import sys

old_excepthook = None

def _except_handler(exc_class, value, tb):
    """
    Called when an uncaught exception is raised.  Depending on the current state
    of the interpreter, either call the old (normal) except hook or drop into 
    the pdb debugger in post mortem mode to enable further analysis.
    
    .. versionadded:: 0.1.0
    """
    
    global old_excepthook
    
    if hasattr(sys, 'ps1') or not sys.stderr.isatty():
        # We aren't in interactive mode or we don't have a tty-like device, so 
        # we call the default hook.
        old_excepthook(exc_class, value, tb)
    
    else:
        import traceback
        import pdb
        
        # We are in interactive mode, print the exception...
        traceback.print_exception(exc_class, value, tb)
        print
        
        # ...then start the debugger in post-mortem mode.
        pdb.pm()

def init():
    """
    Register the new except hook.
    
    .. versionadded:: 0.1.0
    """
    
    # Keep a reference to the old except hook handler so we can use it in :meth:`_except_handler`.
    global old_excepthook
    
    old_excepthook = sys.excepthook
    sys.excepthook = _except_handler

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

