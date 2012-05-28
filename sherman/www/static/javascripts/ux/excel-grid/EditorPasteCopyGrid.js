/**
 * @class Ext.grid.EditorPasteCopyGridPanel
 * Version: 1.4
 * Author: Surinder singh http://www.sencha.com/forum/member.php?75710-Surinder-singh, surinder83singh@gmail.com
 * changes: 1) added the block fill feature.
            2) support for auto editing on any non-navigation key press (feature demanded by jackpan http://www.sencha.com/forum/member.php?181839-jackpan). 
 *
 */
Ext.grid.EditorPasteCopyGridPanel = Ext.extend(Ext.grid.EditorGridPanel, {

     /**
     * @cfg {Number} clicksToEdit
     * <p>The number of clicks on a cell required to display the cell's editor (defaults to 2).</p>
     * <p>Setting this option to 'auto' means that mousedown <i>on the selected cell</i> starts
     * editing that cell.</p>
     */
    clicksToEdit:'auto',
    
    initComponent : function(){
        Ext.grid.EditorPasteCopyGridPanel.superclass.initComponent.call(this);
        /*make sure that selection modal is ExcelCellSelectionModel*/
        this.selModel = new Ext.grid.ExcelCellSelectionModel();
        this.addListener('render',this.addKeyMap, this);
    },  
    addKeyMap:function(){
        var thisGrid = this;
        this.body.on("mouseover", this.onMouseOver, this);
        this.body.on("mouseup", this.onMouseUp, this);
        Ext.DomQuery.selectNode('div[class*=x-grid3-scroller]', this.getEl().dom).style.overflowX='hidden';
        // map multiple keys to multiple actions by strings and array of codes      
        new Ext.KeyMap(Ext.DomQuery.selectNode('div[class*=x-grid3-scroller]', this.getEl().dom).id, [{
            key: "c",
            ctrl:true,
            fn: function(){         
                thisGrid.copyToClipBoard(thisGrid.getSelectionModel().getSelectedCellRange());
            }
        },{
            key: "v",
            ctrl:true,
            fn: function(){                             
                 thisGrid.pasteFromClipBoard();
            }
        }]);
    },
    onMouseOver:function(e){
        this.processEvent("mouseover", e);
    },
    onMouseUp:function(e){
        this.processEvent("mouseup", e);
    }, 
    copyToClipBoard:function(rows){
        this.collectGridData(rows);
        if( window.clipboardData && clipboardData.setData ) {
            clipboardData.setData("text", this.tsvData);
        } else {
            var hiddentextarea = this.getHiddenTextArea();
            hiddentextarea.dom.value = this.tsvData; 
            hiddentextarea.focus();
            hiddentextarea.dom.setSelectionRange(0, hiddentextarea.dom.value.length);           
        }
    },
    collectGridData:function(cr){           
        var row1        = cr[0], col1 = cr[1], row2 = cr[2], col2=cr[3];
        this.tsvData    ="";
        var rowTsv      ="";
        for(var r = row1; r<= row2; r++){
            if(this.tsvData!=""){
                this.tsvData +="\n";
            }
            rowTsv  = "";
            for(var c = col1; c<= col2; c++){
                if(rowTsv!=""){
                    rowTsv+="\t";
                }
                rowTsv += this.store.getAt(r).get( this.store.fields.itemAt(c).name);
            }
            this.tsvData +=rowTsv;
        }
        return this.tsvData;        
    },
        
    pasteFromClipBoard:function(){        
        var hiddentextarea = this.getHiddenTextArea();
        hiddentextarea.dom.value =""; 
        hiddentextarea.focus();
                    
    },  
    updateGridData:function(){
        var Record          = Ext.data.Record.create(this.store.fields.items);          
        var tsvData         = this.hiddentextarea.getValue();        
        tsvData             = tsvData.split("\n");
        var column          = [];
        var cr              = this.getSelectionModel().getSelectedCellRange();
        var nextIndex       = cr[0];
        if( tsvData[0].split("\t").length==1 && ( (tsvData.length==1) || (tsvData.length==2  && tsvData[1].trim()== ""))){//if only one cell in clipboard data, block fill process (i.e. copy a cell, then select a group of cells to paste)
            for( var rowIndex = cr[0]; rowIndex<= cr[2]; rowIndex++){
                for( var columnIndex = cr[1]; columnIndex<= cr[3]; columnIndex++){
                    this.store.getAt(rowIndex).set( this.store.fields.itemAt(columnIndex).name, tsvData[0] );
                }
            }
        }else{                      
            var gridTotalRows   = this.store.getCount();
            for(var rowIndex = 0; rowIndex < tsvData.length; rowIndex++ ){
                if( tsvData[rowIndex].trim()== "" ){
                    continue;
                }
                columns = tsvData[rowIndex].split("\t");
                if( nextIndex > gridTotalRows-1 ){
                    var NewRecord   = new Record({});                   
                    this.stopEditing();                 
                    this.store.insert(nextIndex, NewRecord);                        
                }
                pasteColumnIndex = cr[1];                               
                for(var columnIndex=0; columnIndex < columns.length; columnIndex++ ){
                    this.store.getAt(nextIndex).set( this.store.fields.itemAt(pasteColumnIndex).name, columns[columnIndex] );
                    pasteColumnIndex++;
                }
                nextIndex++;
            }
        }
        this.hiddentextarea.blur();
    },
    getHiddenTextArea:function(){
        if(!this.hiddentextarea){
            this.hiddentextarea = new Ext.Element(document.createElement('textarea'));          
            
            //this.hiddentextarea.setStyle('left','-1000px');
            this.hiddentextarea.setStyle('border','2px solid #ff0000');
            this.hiddentextarea.setStyle('position','absolute');
            //this.hiddentextarea.setStyle('top','-0px');
            this.hiddentextarea.setStyle('z-index','-1');
            this.hiddentextarea.setStyle('width','100px');
            this.hiddentextarea.setStyle('height','30px');
            
            this.hiddentextarea.addListener('keyup', this.updateGridData, this);
            Ext.get(this.getEl().dom.firstChild).appendChild(this.hiddentextarea.dom);
        }
        return this.hiddentextarea;
    }
    
});
Ext.reg('editorPasteCopyGrid', Ext.grid.EditorPasteCopyGridPanel);
