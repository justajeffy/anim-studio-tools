##
#   \namespace  reviewTool.gui.widgets.contextwidget.contextwidget
#
#   \remarks    [desc::commented]
#   
#   \author     Dr. D Studios
#   \date       08/08/11
#

from PyQt4.QtCore           import  pyqtSignal, Qt, QVariant

from PyQt4.QtGui            import  QColor,\
                                    QCursor,\
                                    QMenu,\
                                    QTreeWidget


from .contextitem               import  ContextItem
from .contextversionswidget     import  ContextVersionsWidget
from .versionselectdialog       import  VersionSelectDialog

from ....                       import settings
from ...delegates.griddelegate  import GridDelegate

#--------------------------------------------------------------------------------

class ContextWidget( QTreeWidget ):
    # define signals
    activeStateChanged          = pyqtSignal(object)
    loadingStateChanged         = pyqtSignal()
    versionDoubleClicked        = pyqtSignal(list)
    versionSelectionChanged     = pyqtSignal()
    
    def __init__( self, parent ):
        # initialize the super class
        super(ContextWidget,self).__init__( parent )
        
        # initialize the tree settings
        self.setAlternatingRowColors(   True)
        self.setFocusPolicy(            Qt.StrongFocus )
        self.setSelectionMode(          self.ExtendedSelection )
        self.setSelectionBehavior(      self.SelectItems )
        
        # set header information
        header = self.header()
        header.setClickable(True)
        header.setMovable(False)
        header.setStretchLastSection(False)
        header.setResizeMode(0,header.Stretch)
        header.setMinimumSectionSize(150)
        header.setContextMenuPolicy( Qt.CustomContextMenu )
        
        self.setContextMenuPolicy( Qt.CustomContextMenu )
        
        # sort alphabetically by default
        filters         = settings.departmentFilters()
        departments     = filters.items()
        departments.sort( lambda x,y: cmp( x[1]['name'], y[1]['name'] ) )
        
        # set the column information
        self.setColumnCount( 1 + len(departments) )
        self.setHeaderLabels( ['Entity'] + [ department[1]['name'] for department in departments ] )
        
        # store the department id with the header item so when we move it around, it will
        # update properly
        item = self.headerItem()
        for d, department in enumerate(departments):
            item.setData( d+1, Qt.UserRole, QVariant(department[0]) )
        
        # create the delegate information
        delegate = GridDelegate(self)
        delegate.setShowRows(False)
        self.setItemDelegate(delegate)
        
        # create the versions widget
        self._versionSelectDialog   = VersionSelectDialog(self)
        self._itemsLoading          = []
        self._loading               = False
        self._menu                  = None
        
        # create connections
        header.sectionClicked.connect(              self.selectColumn )
        header.customContextMenuRequested.connect(  self.showHeaderMenu )
        self.itemSelectionChanged.connect(          self.emitVersionSelectionChanged )
        self.itemDoubleClicked.connect(             self.handleItemDoubleClick )
        self.customContextMenuRequested.connect(    self.showMenu )
        
        # set the column information
        self.refreshColumns()
    
    def addContext( self, context ):
        """
                Adds a new context item to the top of the tree
                
                :param      context:
                :type       <Context>:
                
                :return     <ContextItem>:
        """
        # create the context item
        item = ContextItem(context)
        
        # add to the tree
        self.addTopLevelItem(item)
        
        # load based on its visibility
        item.loadEntities()
        
        return item
    
    def activateBetweenExtents( self, dept ):
        """
                Activates the versions between the currently selected, active
                versions based on the inputed department.  So between the
                first checked item and last checked item, all items that are not
                checked will activate the version at the given department.
                
                :param  dept:
                :param  str:
        """
        # collect the selected indexes
        indexes     = self.selectedIndexes()
        dcol        = self.column(dept)
        widgets     = []
        selindexes  = []
        lastsel     = -1
        last        = -1
        
        # determine the limits
        for index in indexes:
            # make sure we have a valid item
            item = self.itemFromIndex(index)
            if ( not item ):
                continue
            
            # make sure there is a context widget at the given column
            widget = self.itemWidget(item,index.column())
            if ( type(widget) != ContextVersionsWidget ):
                continue
            
            # determine if this widget is active
            if ( widget.isActive() ):
                selindexes.append(index)
                last    = len(widgets)
                lastsel = len(selindexes)
                
            elif ( last != -1 ):
                dwidget = self.itemWidget(item,dcol)
                if ( type(dwidget) == ContextVersionsWidget ):
                    selindexes.append(self.indexFromItem( item, dcol ))
                    widgets.append(dwidget)
        
        # go through all the middle widgets, selecting and activating them
        self.blockSignals(True)
        self.setUpdatesEnabled(False)
        
        changed     = False
        selmodel    = self.selectionModel()
        
        # activate the widgets
        for widget in widgets[:last]:
            widget.setActive(True)
            changed = True
        
        # select the indexes
        self.clearSelection()
        for index in selindexes[:lastsel]:
            selmodel.select( index, selmodel.Select )
        
        self.blockSignals(False)
        self.setUpdatesEnabled(True)
        
        if ( changed ):
            self.emitActiveStateChanged(None)
    
    def activateSelection( self ):
        """
                Activates the currently selected versions
                
                :return     <int>: number of items activated
        """
        # collect the selected indexes
        indexes = self.selectedIndexes()
        count   = 0
        
        # disable the ui
        signalsBlocked = self.signalsBlocked()
        self.blockSignals(True)
        
        for index in indexes:
            # make sure we have a valid item
            item = self.itemFromIndex(index)
            if ( not item ):
                continue
            
            # make sure there is a context version widget associated
            # with this item
            widget = self.itemWidget(item,index.column())
            if ( not isinstance( widget, ContextVersionsWidget ) ):
                continue
            
            # activate the widget if it is not already
            if ( not widget.isActive() ):
                widget.setActive(True)
                count += 1
        
        # enable the ui
        self.blockSignals(signalsBlocked)
        
        # emit the state changed if anything has changed
        if ( count ):
            self.emitActiveStateChanged(None)
        
        # return the number of items activated
        return count
    
    def column( self, dept ):
        """
                Returns the index of the column for the inputed department
                by name.
                
                :param      dept:
                :type       <str>:
                
                :return     <int>: -1 if not found, otherwise, value of the column
        """
        item = self.headerItem()
        for i in range(self.columnCount()):
            if ( item.data( i, Qt.UserRole ).toString() == dept ):
                return i
        return -1
    
    def clearCache( self ):
        """
                Clears all the cache for this context widget and resets the information
        """
        # block the ui
        signalsBlocked = self.signalsBlocked()
        self.blockSignals(True)
        self.setUpdatesEnabled(False)
        
        # clear the tree and reset
        self.clear()
        self.refreshColumns()
        
        # enable the ui
        self.setUpdatesEnabled(True)
        self.blockSignals(signalsBlocked)
    
    def departmentAt( self, index ):
        item = self.headerItem()
        return str(item.data(index,Qt.UserRole).toString())
    
    def emitActiveStateChanged( self, versions ):
        if ( not self.signalsBlocked() ):
            self.activeStateChanged.emit(versions)
    
    def emitLoadingStateChanged( self ):
        if ( not self.signalsBlocked() ):
            self.loadingStateChanged.emit()
    
    def emitVersionSelectionChanged( self ):
        if ( not self.signalsBlocked() ):
            self.versionSelectionChanged.emit()
    
    def emitVersionDoubleClicked( self, versions ):
        if ( not self.signalsBlocked() ):
            self.versionDoubleClicked.emit(versions)
    
    def handleItemDoubleClick( self, item, column ):
        widget = self.itemWidget( item, column )
        if ( type(widget) == ContextVersionsWidget ):
            self.emitVersionDoubleClicked( widget.currentVersions() )
    
    def isLoading( self ):
        return self._loading
    
    def markLoading( self, item, state ):
        # mark the item as loading
        if ( state and not item in self._itemsLoading ):
            self._itemsLoading.append(item)
            self.setLoading(len(self._itemsLoading) > 0)
        
        # mark the item as not loading
        elif ( not state and item in self._itemsLoading ):
            self._itemsLoading.remove(item)
            self.setLoading(len(self._itemsLoading) > 0)
    
    def menu( self ):
        return self._menu
    
    def navigateTo( self, code ):
        code = str(code)
        found = None
        for i in range(self.topLevelItemCount()):
            item = self.topLevelItem(i)
            for c in range(item.childCount()):
                child = item.child(c)
                if ( code.startswith( child.text(0) ) ):
                    self.scrollToItem( child )
                    found = child
                    break
                    
            if ( found ):
                break
    
    def pickVersions( self, versions ):
        if ( not versions ):
            return False
        
        point = QCursor.pos()
        self._versionSelectDialog.move((point.x() - self._versionSelectDialog.width()) + 15, point.y() + 5)
        return self._versionSelectDialog.popup(versions)
    
    def refreshColumns( self ):
        """
                Loads the columns for the different departments
        """
        self.blockSignals(True)
        self.setUpdatesEnabled(False)
        
        filters         = settings.departmentFilters()
        
        # reorganize the departments to match the current order
        departments     = filters.items()
        departments.sort( lambda x,y: cmp( x[1]['order'], y[1]['order'] ) )
        
        # update the header data
        header  = self.header()
        count   = self.columnCount() - 1
        
        # update the tree widget header
        for d, department in enumerate(departments):
            code            = department[0]
            index           = self.column(code)
            
            # move the section to match the current ordering
            header.setResizeMode(       index, header.ResizeToContents )
            header.setSectionHidden(    index, not department[1]['enabled'] )
            
            # move the column to the current index
            vindex           = header.visualIndex(index)
            header.moveSection( vindex, d+1 )
            
        self.setUpdatesEnabled(True)
        self.blockSignals(False)
    
    def refreshVersionSorting( self ):
        """
                Refreshes all the sorting for the context items and their entties
        """
        for i in range( self.topLevelItemCount() ):
            item = self.topLevelItem(i)
            for i in range( item.childCount() ):
                item.child(i).reSort()
    
    def refreshUi( self ):
        cols = range(self.columnCount())
        for i in range( self.topLevelItemCount() ):
            item = self.topLevelItem(i)
            for x in range( item.childCount() ):
                child = item.child(x)
                for c in cols:
                    widget = self.itemWidget( child, c )
                    if ( not isinstance( widget, ContextVersionsWidget ) ):
                        continue
                    
                    widget.refresh()
    
    def refreshSelection( self ):
        for item in self.selectedItems():
            item.reload()
    
    def selectColumn( self, column ):
        """
                Select the whole column at once for the given column
                
                :param      column:
                :type       <int>:
        """
        # disable ui
        signalsBlocked = self.signalsBlocked()
        self.blockSignals(True)
        self.setUpdatesEnabled(False)
        
        # clear the current selection
        self.clearSelection()
        
        # set the selection to the indexes in the given column
        selmodel    = self.selectionModel()
        
        # loop through the top level items (context items)
        for i in range( self.topLevelItemCount() ):
            item = self.topLevelItem(i)
            
            # loop through the chil items (entity items)
            for x in range( item.childCount() ):
                entity  = item.child(x)
                index   = self.indexFromItem( entity, column )
                
                # set the selection to the current column
                selmodel.select( index, selmodel.Select )
                
                # force the loading of the versions on the entity
                # immediately (to start the thread to collect versions
                # if the entity is not in the visible area)
                entity.loadVersions(True)
        
        # enable ui
        self.setUpdatesEnabled(True)
        self.blockSignals(signalsBlocked)
        
        # emit the selection changed signal
        self.emitVersionSelectionChanged()
    
    def selectedVersions( self, activeOnly = False ):
        """
                Returns the selected versions for this context widget.
                If the activeOnly flag is set to True, then only the active
                versions that are selected will be returned, otherwise, all
                currently selected versions will be returned.
                
                :param      activeOnly:
                :type       <bool>:
                
                :return     <list> [ <Version>, .. ]:
        """
        # collect all the selected indexes
        indexes = self.selectedIndexes()
        
        # generate the output list
        output  = []
        
        for index in indexes:
            # make sure we have a valid index
            item = self.itemFromIndex(index)
            if ( not item ):
                continue
            
            # make sure that the index has versions associated with it
            widget = self.itemWidget(item,index.column())
            if ( not isinstance( widget, ContextVersionsWidget ) ):
                continue
            
            # collect the output information
            if ( activeOnly ):
                output += widget.activeVersions()
            else:
                output += widget.currentVersions()
        
        # return the output
        return output
    
    def setActiveVersions( self, versions ):
        """
                Goes through and activates the versions for all the indexes
                based on the inputed versions
                
                :param      versions:
                :type       <list> [ <Version>, .. ]
        """
        cols        = range(self.columnCount())
        
        for i in range( self.topLevelItemCount() ):
            item = self.topLevelItem(i)
            
            for x in range( item.childCount() ):
                child = item.child(x)
                
                for c in cols:
                    widget = self.itemWidget( child, c )
                    if ( not isinstance( widget, ContextVersionsWidget ) ):
                        continue
                    
                    widget.setActiveVersions(versions)
    
    def setContexts( self, contexts ):
        """
                Initializes all the default context instances that this context widget
                will display.  It will clear the current tree, and add the inputed contexts
                as hidden tree items in the widget.
                
                :param      contexts:
                :type       <list> [ <Context>, .. ]:
        """
        # disable the ui
        signalsBlocked = self.signalsBlocked()
        self.blockSignals(True)
        self.setUpdatesEnabled(False)
        
        # clear the current contents
        self.clear()
        
        # add the context items to the tree
        for context in contexts:
            item = ContextItem(context)
            self.addTopLevelItem(item)
            item.setHidden(True)
        
        # enable the ui
        self.setUpdatesEnabled(True)
        self.blockSignals(signalsBlocked)
    
    def setMenu( self, menu ):
        self._menu = menu
    
    def setVisibleContexts( self, contexts, focusEntity = None ):
        """
                Sets the contents for this widget to the given context list
                
                :param  contexts:
                :type   <list> [ <Context>, .. ]
                
                :param  focusEntity:    entity to be focused on when loading
                :type   <str> || None:
        """
        # disable the ui
        signalsBlocked = self.signalsBlocked()
        self.blockSignals(True)
        
        # loop through the tree, looking for the contexts
        # to show
        for i in range(self.topLevelItemCount()):
            item = self.topLevelItem(i)
            
            # make sure the item is visible and loaded
            if ( item.context() in contexts ):
                item.setHidden(False)
                
                # initialize the entity items (if not already done so)
                item.loadEntities(focusEntity)
            else:
                item.setHidden(True)
        
        # enable the ui
        self.blockSignals(signalsBlocked)
        
    def setLoading( self, state = True ):
        """
                Marks this widget as being in a loading state.  If the state matches the
                current state, then nothing happens, otherwise, the state switches and the
                loadingStateChanged signal is emitted (as long as this widget does not currently
                have its signals blocked).
                
                :param      state:
                :type       <bool>:
                
                :return     <bool>: changed
        """
        # check to see if the state is actually changing
        if ( self._loading == state ):
            return False
        
        # update the loading state and emit the signal
        self._loading = state
        self.emitLoadingStateChanged()
        return True
    
    def setSelectedVersions( self, selected ):
        """
                Sets the list of selected versions to the inputed list of versions
                
                :param      selected:
                :type       <list> [ <Version>, .. ]:
        """
        # disable the ui
        signalsBlocked = self.signalsBlocked()
        self.blockSignals(True)
        self.setUpdatesEnabled(False)
        
        # create 
#        selection_set   = set(selected)
        colrange        = range(self.columnCount())
        selmodel        = self.selectionModel()
        first           = True
        
        # clear the current selection
        self.clearSelection()
        
        # loop through all the top level items (context items)
        for i in range( self.topLevelItemCount() ):
            item = self.topLevelItem(i)
            
            # loop through all the child items (entity items)
            for x in range( item.childCount() ):
                entity  = item.child(x)
                
                # loop through all the columns (version widgets)
                for c in colrange:
                    
                    # grab the widget for the current entity and column
                    widget = self.itemWidget(entity,c)
                    
                    # make sure we have a valid context version widget
                    if ( not isinstance( widget, ContextVersionsWidget ) ):
                        continue
                    
                    # update the widget (based on the current active versions)
                    widget.refresh()
                    
                    # if there is overlap between the inputed selection and active versions
                    # of the widget, then select the index
                    found = False
                    for ver in widget.activeVersions():
                        if ( ver in selected ):
                            found = True
                            break
                    
                    if ( not found ):
                        continue
                        
                    selmodel.select( self.indexFromItem( entity, c ), selmodel.Select )
                    
                    if ( first ):
                        self.scrollToItem(entity)
                        first = False
                        
        
        # enable the ui
        self.blockSignals(signalsBlocked)
        self.setUpdatesEnabled(True)
        
        # emit the selection changed signal
        self.emitVersionSelectionChanged()
    
    def showHeaderMenu( self ):
        depts   = settings.departments()
        labels  = settings.departmentLabels(depts)
        visible = settings.enabledDepartments()
        
        # create the menu
        menu = QMenu(self)
        for d, dept in enumerate(depts):
            act = menu.addAction(labels[d])
            act.setObjectName(dept)
            act.setCheckable(True)
            act.setChecked( dept in visible )
        
        menu.triggered.connect( self.toggleDeptTriggered )
        
        menu.exec_(QCursor.pos())
    
    def showMenu( self ):
        if ( self._menu ):
            self._menu.exec_(QCursor.pos())
    
    def toggleDeptTriggered( self, action ):
        dept = str(action.objectName())
        settings.enableDepartment(dept,action.isChecked())
        self.refreshColumns()
    
    def versionsAvailable( self ):
        """
                Returns whether there are versions available for playing.  This check is
                slightly faster than the selection lookup, as it only has to find a singular
                version to play vs. looking through all the selected indexes.
                
                :return     <bool>: found
        """
        # collect the selected indexes
        indexes     = self.selectedIndexes()
        count       = 0
        
        for index in indexes:
            # make sure the item is valid and visible
            item = self.itemFromIndex(index)
            if ( not item or item.isHidden() ):
                continue
            
            # if any item is loading, then mark the versions available as
            # false.  Need to wait for all selected versions to finish loading
            # before processing them
            elif ( item.isLoading() ):
                count = 0
                break
            
            # make sure there are versions associated with this index
            widget = self.itemWidget(item,index.column())
            if ( not isinstance( widget, ContextVersionsWidget ) ):
                continue
            
            # increment the valid number of versions count
            count += 1
        
        return count > 0
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

