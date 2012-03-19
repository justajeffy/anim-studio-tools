#        Dr. D Studios - Software Disclaimer
#
# Copyright 2010 Dr D Studios Pty Limited (ACN 127 184 954) (Dr. D Studios), its
# affiliates and/or its licensors.
#
# The software, information, code, data and other materials (Software)
# contained in, or related to, these files is the confidential and proprietary
# information of Dr. D Studios, its affiliates and/or its licensors. The
# Software is protected by copyright. The Software must not be disclosed,
# distributed or provided to any third party without the prior written
# authorisation of Dr. D Studios.
#
# Dr. D Studios makes no express warranties in relation to the Software. The
# Software is supplied 'as is' and to the fullest extent permitted by law,
# all statutory or implied conditions and warranties are excluded, including,
# without limitation, any implied warranties of merchantability or fitness for
# a particular purpose, and warranties as to accuracy, completeness or adequacy
# of information. Dr. D Studios does not warrant that the Software is free from
# defects, will operate in an uninterrupted or error free manner or that the
# Software will work with any or all operating systems.
#
# To the extent permitted by law, in no event will Dr. D Studios, its
# affiliates, suppliers or licensors be liable for any direct or indirect,
# special, punitive or consequential loss or damage even if such loss or damage
# was reasonably foreseeable including, without limitation, loss of revenue,
# profits, goodwill, bargain, anticipated savings or loss or corruption of
# data, arising out of or relating to the use of the Software.

"""
.. versionadded:: 1.0

Extension that adds ability to auto-generate a code listing in the page.
"""

from docutils import nodes, utils
from sphinx.util import split_explicit_title
from sphinx.util.compat import Directive
from sphinx import addnodes

        
class autolisting_node(nodes.General, nodes.Element):
    """
    Place holder node - replaced by a list
    """
    pass
    
    
class AutolistingDirective(Directive):
    """
    The directive that inserts the placeholder node.
    """
    
    def run(self):
        return [autolisting_node()]


def collect_items_for_autolisting(doctree):
    """
    Collect class and function references in doctree
    
    Return list in order encountered
    """
    items = [] 
    
    def condition(n):
        return isinstance(n, addnodes.desc) and n.attributes.get('desctype', None) in ('class', 'exception', 'function', 'method')
    
    for node in doctree.traverse(condition=condition):
        desctype = node.attributes.get('desctype')
        sig = node.children[node.first_child_matching_class(addnodes.desc_signature)]
        name = sig.attributes.get('names')[0]
        id = sig.attributes.get('ids')[0]
        items.append((desctype, name, id))
    return items


def generate_autolisting(items):
    """
    Generate nodes for autolisting that should replace the placeholder node (autolisting_node)
    
    :param items: list of tuples in form (type, name, id)
    :type items: list
    
    Example output::
    
        Classes
            * ClassA 
                * member_1
                * member_2
            * ClassB 
                * member_1
                * member_2
            
        Functions
            * FunctionA
            * FunctionB
    
        Exceptions
            * ExceptionA
            * ExceptionB
    
    """
    classes = {}
    functions = {}
    exceptions = {}
    
    for typ, name, id in items:
        print typ, name, id
        if typ == "class":
            classes[name] = {"name":name, "id":id, "methods":{}}
        elif typ == "exception":
            exceptions[name] = {"name":name, "id":id, "methods":{}}
        elif typ == "method":
            class_name, method_name = name.rsplit(".", 1)
            if class_name in classes:
                classes[class_name]["methods"][method_name] = {"name":method_name, "id":id}
            elif class_name in exceptions:
                exceptions[class_name]["methods"][method_name] = {"name":method_name, "id":id}
        elif typ == "function":
            functions[name] = {"name":name, "id":id}
    
    contents = []
    
    # classes
    if classes:
        classes_title = nodes.emphasis()
        classes_title += nodes.Text('Classes')
        contents.append(classes_title)
        
        bl = nodes.bullet_list()
        for class_name, data in classes.items():
            li = nodes.list_item()
            para = nodes.paragraph()
    
            ref = nodes.reference(refid=data["id"], reftitle=data["name"])
            lit = nodes.literal(classes=["xref"])
            lit += nodes.Text(data["name"])
            ref += lit
            para += ref
            li += para
            bl += li     
            
            if len(data["methods"]):
                sbl = nodes.bullet_list()
                for method_name, meth_data in data["methods"].items():
                    sli = nodes.list_item()
                    spara = nodes.paragraph()
                    ref = nodes.reference(refid=meth_data["id"], reftitle=meth_data["name"])
                    lit = nodes.literal(classes=["xref"])
                    lit += nodes.Text(meth_data["name"])
                    ref += lit
                    spara += ref
                    sli += spara
                    sbl += sli
                bl += sbl
            
        contents.append(bl)
    
    # functions
    if functions:
        functions_title = nodes.emphasis()
        functions_title += nodes.Text('Functions')
        contents.append(functions_title)
        
        bl = nodes.bullet_list()
        for function_name, data in functions.items():
            li = nodes.list_item()
            para = nodes.paragraph()
            
            ref = nodes.reference(refid=data["id"], reftitle=data["name"])
            lit = nodes.literal(classes=["xref"])
            lit += nodes.Text(data["name"])
            ref += lit
            para += ref
            li += para
            bl += li
            
        contents.append(bl)    
    
    # exceptions
    if exceptions:
        exceptions_title = nodes.emphasis()
        exceptions_title += nodes.Text('Exceptions')
        contents.append(exceptions_title)
        
        bl = nodes.bullet_list()
        for exception_name, data in exceptions.items():
            li = nodes.list_item()
            para = nodes.paragraph()
    
            ref = nodes.reference(refid=data["id"], reftitle=data["name"])
            lit = nodes.literal(classes=["xref"])
            lit += nodes.Text(data["name"])
            ref += lit
            para += ref
            li += para
            bl += li     
            
            if len(data["methods"]):
                sbl = nodes.bullet_list()
                for method_name, meth_data in data["methods"].items():
                    sli = nodes.list_item()
                    spara = nodes.paragraph()
                    ref = nodes.reference(refid=meth_data["id"], reftitle=meth_data["name"])
                    lit = nodes.literal(classes=["xref"])
                    lit += nodes.Text(meth_data["name"])
                    ref += lit
                    spara += ref
                    sli += spara
                    sbl += sli
                bl += sbl
            
        contents.append(bl)
    
    return contents


def auto_list(app, doctree, fromdocname):
    """
    Generate a listing for classes, methods and functions described in the file
    """
    env = app.builder.env

    for node in doctree.traverse(autolisting_node):
        content = []
        
        sidebar = nodes.sidebar()
        
        para = nodes.paragraph()
        part_title = nodes.strong()
        part_title += nodes.Text('Code Listing')
        para += part_title
        sidebar += para
        
        items = collect_items_for_autolisting(doctree)
        listings = generate_autolisting(items)

        for entry in listings:
            sidebar += entry
            
        # only show sidebar if sidebar has content
        if listings:
            content.append(sidebar)
            
        node.replace_self(content)
        break
    

def setup(app):
    """
    Add autolisting to current Sphinx app instance
    """

    app.add_node(autolisting_node)
    app.add_directive('autolisting', AutolistingDirective)
    app.connect('doctree-resolved', auto_list)

    
    
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

