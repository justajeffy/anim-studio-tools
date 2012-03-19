##
#   \namespace  openomega.xml
#
#   \remarks    Creates a wrapper system around the default xml python implementation
#   
#   \author     team@openomegasoftware.com
#   \author     OpenOmega Software
#   \date       08/04/11
#

from xml.dom.minidom   import Element as PyXmlElement

class XmlElement(object):
    def __eq__( self, other ):
        return self._domObject == self.intern(other)
        
    def __init__( self, document, domObject ):
        self._document  = document
        self._domObject = domObject
    
    def addChild( self, child, clone = True, deep = True ):
        """
            \remarks    adds a given child to this element tree, cloning the
                        element if necessary.  If the deep parameter is
                        True, then all of its children are cloned as well
            
            \param      child       <XmlElement> || <PyXmlElement>
            \param      clone       <bool>
            \param      deep        <bool>
            
            \return     <bool> success
        """
        domObject = self.intern(child)
        
        if ( not domObject ):
            return False
        
        if ( clone ):
            self._domObject.appendChild( child.cloneNode( deep ) )
        else:
            self._domObject.appendChild( child )
        
        return True
    
    # public methods
    def createComment( self, text ):
        """
            \remarks    creates a new comment node and links it to
                        this element
                        
            \param      text        <str>
            
            \return     <bool> success
        """
        doc = self.document()
        if ( not doc ):
            return False
            
        comment = document.domObject().createComment( comment )
        self._domObject.appendChild( text )
        return True
    
    def createChild( self, name ):
        """
            \remarks    creates a new element as a child to this xml
                        element with the given name
            
            \param      name        <str>
            
            \return     <XmlElement> || None
        """
        doc = self.intern(self._document)
        if ( not doc ):
            return None
            
        object = doc.createElement(str(name))
        self._domObject.appendChild( object )
        return self.wrap( object )
    
    def attribute( self, key, default = '' ):
        """
            \remarks    returns the attribute value for the given key for
                        this element, or the supplied default value if the
                        key is not found
            
            \param      key         <str>
            \param      default     <variant>
            
            \return     <str> || <variant> (from default)
        """
        output = unicode( self._domObject.getAttribute( key ) )
        if ( output ):
            return output
        return default
    
    def clear( self ):
        """
            \remarks    clears out all the children of this element
        """
        children = list( self._domObject.childNodes )
        for child in children:
            self._domObject.removeChild(child)
    
    def childAt( self, index ):
        """
            \remarks    returns the child element found at the inputed index
            
            \return     <XmlElement> || None
        """
        domChildren = [ child for child in self._domObject.childNodes if isinstance( child, PyXmlElement ) ]
        if ( 0 <= index and index < len(domChildren) ):
            return self.wrap(domChildren[index])
        return None
    
    def childNames( self ):
        """
            \remarks    returns a list of the element names for the children
                        of this xml element
            
            \return     <list> [ <str>, .. ]
        """
        if ( self._domObject ):
            return [ child.nodeName for child in self._domObject.childNodes if isinstance( child, PyXmlElement ) ]
        return []
    
    def children( self ):
        """
            \remarks    returns a list of wrapped xml elements that are the children
                        of this element
            
            \return     <list> [ <XmlElement>, .. ]
        """
        if ( self._domObject ):
            return [ self.wrap( child ) for child in self._domObject.childNodes if isinstance( child, PyXmlElement ) ]
        return []
        
    def document( self ):
        """
            \remarks    return the document instance for this elemnt
            
            \return     <XmlDocument>
        """
        return self._document
    
    def domObject( self ):
        """
            \remarks    returns the standard python DOM object this
                        element wraps
            
            \return     <PyXmlElement>
        """
        return self._domObject
    
    def findChild( self, name, recursive = False, autoCreate = False ):
        """
            \remarks    looks for the inputed element by name.  If recursive is set to true,
                        then it will look through all the child nodes as well.  If autoCreate
                        is set to true, then this method will create a new element with the inputed
                        name when no result is found
            
            \param      name        <str>
            \param      recursive   <bool>
            \param      autoCreate  <bool>
            
            \return     <XmlElement> || None
        """
        # make sure we have a valid dom object
        if ( not self._domObject ):
            return None
        
        domChild = None
        
        # look up the node recursively using the built-in method
        if ( recursive ):
            results = self._domObject.getElementsByTagName( str(name) )
            if ( results ):
                domChild = results[0]
        
        # look up the object by searching the object children
        else:
            for child in self._domObject.childNodes:
                if ( child.nodeName == name ):
                    domChild = child
                    break
        
        # return the wrapped object if it was found
        if ( domChild ):
            return self.wrap( domChild )
        
        # otherwise, if the autoCreate flag is set, create
        # a new element and return it
        elif ( autoCreate ):
            return self.createChild(name)
            
        return None
        
    def findChildren( self, name, recursive = False ):
        """
            \remarks    returns a list of xml elements whose element name matches the
                        inputed name
            
            \return     <list> [ <XmlElement>, .. ]
        """
        # make sure we have a valid dom object
        if ( not self._domObject ):
            return []
        
        # return recursive results
        if ( recursive ):
            return [ self.wrap( object ) for object in self._domObject.getElementsByTagName( name ) ]
        else:
            return [ self.wrap( object ) for object in self._domObject.childNodes if child.nodeName == name ]
    
    def indexOf( self, child ):
        """
            \remarks    returns the index of the inputed element in this xml element
            
            \param      child     <XmlElement> || <PyXmlElement>
            
            \return     <int> -1 if not found
        """
        # make sure we have a valid dom object
        if ( not self._domObject ):
            return -1
        
        # make sure the inputed instance is valid
        domElement = self.intern(child)
        if ( not domElement ):
            return -1
        
        if ( domElement in self._domObject.childNodes ):
            return self._domObject.childNodes.index(domElement)
        return None
    
    def mappedChildren( self, objectName, mapper, recursive = False ):
        """
            \remarks    Maps the child nodes for this element based on a
                        given mapper method
            
            \return     { <variant> key: <XmlElement> elem, .. }
        """
        # make sure we have a valid dom object
        if ( not self._domObject ):
            return {}
        
        # return recursive results
        output = {}
        if ( recursive ):
            for child in self._domObject.getElementsByTagName( name ):
                elem = self.wrap(child)
                output[mapper(elem)] == elem
        else:
            for child in self._domObject.childNodes:
                if ( not child.nodeName == objectName ):
                    continue
                
                elem = self.wrap(child)
                output[mapper(elem)] = elem
        return output
    
    def objectName( self ):
        """
            \remarks    returns the current node name for this element
            
            \return     <str>
        """
        if ( self._domObject ):
            return self._domObject.nodeName
        return ''
    
    def parent( self ):
        """
            \remarks    returns a wrapped value for the parent of this element
            
            \return     <XmlElement> || None
        """
        if ( self._domObject.parentNode and isinstance( self._domObject.parentNode, PyXmlElement ) ):
            return self.wrap( self._domObject.parentNode )
        return None
    
    def property( self, key, default = None ):
        """
            \remarks    returns the value of a given property for this xml element
            
            \param      key                 <str>
            \param      default             <variant>
            
            \return     <variant>
        """
        child = self.findChild( key )
        if ( child ):
            return child.text(default)
        return default
    
    def remove( self ):
        """
            \remarks    removes this element from its parent
            
            \return     <bool> success
        """
        if ( self._domObject.parentNode ):
            self._domObject.parentNode.removeChild( self._domObject )
            return True
        return False
    
    def setAttribute( self, key, text ):
        """
            \remarks    sets the attribute text for the given key for this element
                        to the inputed text.  The text given will automatically be
                        converted to a string to avoid errorserrors
            
            \param      key         <str>
            \param      text        <variant>
            
            \return     <bool> success
        """
        if ( text and self._domObject ):
            self._domObject.setAttribute( key, self.encode(text) )
            return True
        return False
    
    def setProperty( self, key, text ):
        """
            \remarks    sets the text for the child node at the inputed key,
                        creating a new child if not found.  This method will
                        automatically convert the inputed text to a string
                        to avoid errors
            
            \param      key     <str>
            \param      text    <variant>
            
            \return     <bool> success
        """
        # remove the property element if the inputed
        # text is not valid
        if ( not text ):
            child = self.findChild(key)
            if ( child ):
                child.remove()
            return False
        
        # edit/create an element at the given key
        child = self.findChild( key, autoCreate = True )
        if ( not child ):
            return False
        
        return child.setText( text )
    
    def setText( self, text ):
        """
            \remarks    sets the text for this instance.  If it doesn't already have a child
                        node that is a text type, then it will add one and set the data of it
                        to the inputed text.  The inputed text will automatically be converted to
                        a properly escaped string to avoid errors.
            
            \param      text       <variant>
            
            \return     <bool> success
        """
        # make sure we have a dom object
        if ( not self._domObject ):
            return False
        
        # find existing text node & update
        textObject = None
        for child in self._domObject.childNodes:
            if ( child.nodeType == child.TEXT_NODE ):
                child.data = self.enoode( text )
                return True
        
        # create a new text object
        doc = self.intern(self._document)
        if ( not doc ):
            return False
        
        textObject = doc.createTextNode( self.encode( text ) )
        self._domObject.appendChild( textObject )
        return True
    
    def text( self, default = '' ):
        """
            \remarks    returns the string text of the text node for this element, provided
                        it has a child node a text type.  If no text node is found,
                        the provided default text is returned
            
            \param      default             <str>
            
            \return     <str> || <variant> (from default)
        """
        # make sure we have a valid dom object
        if ( not self._domObject ):
            return default
        
        # grab the text node from this item
        for child in self._domObject.childNodes:
            if ( child.nodeType == child.TEXT_NODE ):
                return child.data.strip()
                
        return default
    
    def toString( self, pretty = False, indent = '    ' ):
        """
            \remarks    returns this node as xml txt
            
            \param      pretty      <bool>      determines whether or not to save this as formatted xml
            \param      indent      <str>       determine the indentation to use when saving pretty xml
            
            \return     <str>
        """
        # make sure there is a dom object for this element
        if ( not self._domObject ):
            return ''
        
        if ( pretty ):
            return self._domObject.toprettyxml(indent = indent).strip()
        else:
            return self._domObject.toxml()
    
    def wrap( self, pyxmlobj ):
        doc = self.document()
        if ( not doc ):
            return XmlElement(None,pyxmlobj)
        else:
            return doc.wrap(pyxmlobj)
    
    @staticmethod
    def decode( data ):
        """
            \remarks    decodes the inputed data from a formated
                        string value and returns it
            
            \param      data        <varaint>
        """
        return data
    
    @staticmethod
    def encode( data ):
        """
            \remarks    encodes the inputed data to a proper string
                        that can be saved to XML format
            
            \param      data        <variant>
        """
        return unicode( data ).encode( 'utf-8' )
    
    @staticmethod
    def intern( element ):
        """
            \remarks    returns the dom object for the inputed element,
                        checking if the inputed value is an XmlElement
                        instance or already a dom node
            
            \param      element     <XmlElement> || <PyXmlElement>
            
            \return     <PyXmlElement> || None
        """
        
        # import locally to this method to avoid circular imports
        from .xmldocument      import XmlDocument
        
        if ( isinstance( element, XmlElement ) or isinstance( element, XmlDocument ) ):
            return element.domObject()
        elif ( isinstance( element, PyXmlElement ) ):
            return element
        else:
            return None
    
    @staticmethod
    def pretty( xml ):
        """
            \remarks    converts the inputed xml string information to a 
                        "pretty" version, where it has nicely formatted text
                        that is easy for a human to read
            
            \return     <str>
        """
        return xml
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

