##
#   \namespace  openomega.xml.xmlpatch
#
#   \remarks    Patches the base xml dom classes with preferrable updates
#   
#   \author     team@openomegasoftware.com
#   \author     OpenOmega Software
#   \date       08/04/11
#

from xml.dom.minidom                import  Document as BaseDocument,\
                                            Element  as BaseElement,\
                                            Text     as BaseText,\
                                            _write_data

class Document(BaseDocument):
    def writexml(self, writer, indent="", addindent="", newl="", encoding = None):
        """
            \remarks    overloads the base xml.dom.minidom.Document
                        writexml method to make the pretty xml generation
                        a little prettier - doesn't create endlines after
                        every text node.
            
            \param      writer      <file> || <stream>
            \param      indent      <str>               current indent level
            \param      addindent   <str>               indentation to addindent
            \param      newl        <str>               new line character
            \param      encoding    <str> || None
            
            \return     <str>
        """
        # write the base xml information
        if encoding is None:
            writer.write('<?xml version="1.0" ?>')
        else:
            writer.write('<?xml version="1.0" encoding="%s"?>' % (encoding))
        
        # write the chil nodes
        for node in self.childNodes:
            node.writexml(writer, indent, addindent, newl)
            
#--------------------------------------------------------------------------------

class Element(BaseElement):
    def writexml(self, writer, indent="", addindent="", newl=""):
        """
            \remarks    overloads the base xml.dom.minidom.Element
                        writexml method to make the pretty xml generation
                        a little prettier - doesn't create endlines after
                        every text node.
            
            \param      writer      <file> || <stream>
            \param      indent      <str>               current indent level
            \param      addindent   <str>               indentation to addindent
            \param      newl        <str>               new line character
            
            \return     <str>
        """
        writer.write('%s%s<%s' % (newl,indent,self.tagName))
        
        # collect and record the attributes by name
        attrs = self._get_attributes()
        a_names = attrs.keys()
        a_names.sort()
        
        # record the attributes
        for a_name in a_names:
            writer.write(' %s="' % a_name)
            _write_data(writer, attrs[a_name].value)
            writer.write('"')
            
        # record the children
        if self.childNodes:
            writer.write('>')
            for node in self.childNodes:
                if ( node.nodeType == Element.TEXT_NODE ):
                    node.writexml( writer, '', '', '' )
                else:
                    node.writexml( writer, indent + addindent, addindent, newl )
                
                lastnodetype = node.nodeType
                
            if ( lastnodetype == Element.TEXT_NODE ):
                writer.write('</%s>' % self.tagName)
            else:
                writer.write( '%s%s</%s>' % (newl,indent,self.tagName) )
        else:
            writer.write('/>')
            
#--------------------------------------------------------------------------------

class Text(BaseText):
    def writexml( self, writer, indent="", addindent="", newl="" ):
        """
            \remarks    overloads the base xml.dom.minidom.Document
                        writexml method to make the pretty xml generation
                        a little prettier - doesn't create endlines after
                        every text node.
            
            \param      writer      <file> || <stream>
            \param      indent      <str>               current indent level
            \param      addindent   <str>               indentation to addindent
            \param      newl        <str>               new line character
            
            \return     <str>
        """
        _write_data(writer, '%s' % self.data)
        
#--------------------------------------------------------------------------------

# reset the minidom classes
import xml.dom.minidom
xml.dom.minidom.Document    = Document
xml.dom.minidom.Element     = Element
xml.dom.minidom.Text        = Text
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

