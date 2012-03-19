
"""
.. versionadded:: 1.0

Extension that adds ability to reference external links defined in config file.
"""

from docutils import nodes, utils
from sphinx.util import split_explicit_title
from sphinx.util.compat import Directive
from sphinx import addnodes


def ext_ref_role(typ, rawtext, text, lineno, inliner, options={}, content=[]):
    """
    A role that allows referencing external links defined in the config file.
    This allows urls to be easily updated without changing all the documentation.
    
    Define 'external_links' as a dictionary in your config file and link to using `:extref:`referance_name``
    If the reference text contains a forward slash it will be split and the first part used as reference, with the 
    second part appended to the generated link.
    
    Example::
        
        # conf.py
        # ---------------
        external_links = {"Rodin": "http://drddocs/drd/software/int/apps/Rodin/v00_00_09/docs/_build/html/" }
        
        
        # intro.rst
        .. seealso:: 
            
            :extref:`Rodin Documentation <Rodin>`
            :extref:`Rodin User Docs <Rodin/user>`
            
    """
    env = inliner.document.settings.env
    text = utils.unescape(text)
    
    # split any title, link
    has_explicit_title, title, target = split_explicit_title(text)
    
    # check if target contains a slash
    extra = None
    if "/" in target:
        target, extra = target.split("/", 1)
    
    # now find the linked target
    link = env.config.external_links.get(target, None)
    if not link:
        msg = inliner.reporter.warning("No external link specified in config for target '%s'" % target, line=lineno)
        prb = inliner.problematic(rawtext, rawtext, msg)
        return [prb], [msg]
    else:
        # display a uri link reference with given title
        if extra:
            link = "%s/%s" % (link, extra)
            
        pnode = nodes.reference(title, title, refuri=link)
        pnode.lineno = lineno
        return [pnode], []
    

def setup(app):
    """
    Add external reference directive to current Sphinx app instance
    """
    app.add_config_value('external_links', {}, False)
    app.add_role('extref', ext_ref_role)

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

