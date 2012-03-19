##
#   \namespace  openomega.xml.xmldocument
#
#   \remarks    Creates a simple wrapper class for the xml.dom.minidom.Document
#               class type for convenient usages
#   
#   \author     team@openomegasoftware.com
#   \author     OpenOmega Software
#   \date       08/05/11
#

import os

import xml.dom.minidom
from xml.dom.minidom            import Document as PyXmlDocument

from .xmlelement   import XmlElement

class XmlDocument(object):
    def __init__( self ):
        self._domObject = PyXmlDocument()
        self._filename  = ''
        self._elementClasses = {}
    
    def clear( self ):
        """
            \remarks    clears the information for this document
            
            \return     <bool> success
        """
        # make sure we have a valid object
        if ( self._domObject ):
            return False
        
        children = list( self._domObject.childNodes )
        for child in children:
            self._domObject.removeChild( child )
        
        return True
    
    def elementClass( self, objectName = '' ):
        return self._elementClasses.get(objectName)
    
    def domObject( self ):
        """
            \remarks    returns the DOM object associated with this
                        document
            
            \return     <PyXmlDocument> || None
        """
        return self._domObject
    
    def fromString( self, xml ):
        """
            \remarks    parses the inputed xml string into an XmlElement hierarchy
            
            \param      xml     <str>
            
            \return     <bool> success
        """
        # convert the xml to a valid string
        xml = unicode( xml ).encode('utf-8')
        if ( not xml ):
            return False
        
        try:
            domObject = xml.dom.minidom.parseString( xml )
        except Exception, e:
            return False
        
        # make sure we have a valid object
        if ( not domObject ):
            return False
        
        self._domObject = domObject
        self._filename  = ''
        return True
    
    def load( self, filename ):
        """
            \remarks    loads the given xml file by calling the python xml parse method, setting this instances
                        dom object to the resulting value
            
            \param      filename        <str>
            
            \return     <bool> success
        """
        filename = str(filename)
        
        # log the error if the filename doesn't exist
        if ( not os.path.exists( filename ) ):
            return False
        
        # try to parse the file
        try:
            domObject   = xml.dom.minidom.parse( filename )
        except Exception, e:
            return False
        
        # make sure we have a valid dom object
        if ( not domObject ):
            return False
        
        self._domObject = domObject
        self._filename  = filename
        return True
    
    def root( self ):
        """
            \remarks    returns the root node for this document, or None if none was found
            
            \return     <XmlElement> || None
        """
        if ( self._domObject and self._domObject.childNodes ):
            return self.wrap( self._domObject.childNodes[0] )
        return None
    
    def setElementClass( self, elementClass, objectName = '' ):
        self._elementClasses[objectName] = elementClass
    
    def setRoot( self, element ):
        """
            \remarks    sets the root node for this document using the inputed
                        value.  The element input can be a PyXmlElement, XmlElement
                        or string value.
            
            \param      element   <str> || <XmlElement> || <PyXmlElement>
            
            \return     <XmlElement> || None
        """
        # make sure we have a valid document
        if ( not self._domObject ):
            return None
        
        # clear the current document hierarchy, as XML documents can have only
        # 1 root node
        self.clear()
        
        # try to convert this object to a PyXmlElement
        root = XmlElement.intern(element)
        if ( not root ):
            root = self._domObject.createElement(str(element))
        
        domObject = self._domObject.appendChild(root)
        return self.wrap(domObject)
    
    def save( self, filename, pretty = True, indent = '    ' ):
        """
            \remarks    saves the xml document to the inputed file.  If the pretty flag is
                        supplied, then it will save it with proper human readable spacing,
                        otherwise it will just try to maximize space.
            
            \param      filename    <str>
            \param      pretty      <bool>
            \param      indent      <str>
            
            \return     <bool> success
        """
        filename = str(filename)
        path     = os.path.dirname( filename )
        # make sure the directory exists
        if ( not os.path.exists( path ) ):
            return False
        
        # format the xml data
        text = self.toString( pretty = pretty, indent = indent )
        
        # create the file and save it
        f = open( filename, 'w' )
        f.write( text )
        f.close()
        
        self._filename = filename
        
        return True
    
    def toString( self, pretty = False, indent = '    ' ):
        """
            \remarks    converts the document to string information
            
            \param      pretty      <bool>
            \param      indent      <str>
            
            \return     <str>
        """
        # make sure we have a dom object
        if ( not self._domObject ):
            return ''
            
        if ( pretty ):
            return self._domObject.toprettyxml( indent = indent ).strip()
        else:
            return self._domObject.toxml()
    
    def wrap( self, pyxmlobj ):
        cls = self._elementClasses.get(pyxmlobj.nodeName)
        if ( cls ):
            return cls(self,pyxmlobj)
        
        cls = self._elementClasses.get('')
        if ( cls ):
            return cls(self,pyxmlobj)
        
        return XmlElement(self,pyxmlobj)
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

