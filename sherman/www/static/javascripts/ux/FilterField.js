/*!
 * Ext JS Library 3.0.0
 * Copyright(c) 2006-2009 Ext JS, LLC
 * licensing@extjs.com
 * http://www.extjs.com/license
 */
Ext.ns('Ext.ux.form');

Ext.ux.form.FilterField = Ext.extend(Ext.form.TwinTriggerField, {
    initComponent : function(){
        Ext.ux.form.FilterField.superclass.initComponent.call(this);
        this.on('specialkey', function(f, e){
            if(e.getKey() == e.ENTER){
                this.onTrigger3Click();
            }
        }, this);
    },

    validationEvent:false,
    validateOnBlur:false,
    trigger1Class:'x-form-clear-trigger',
    trigger2Class:'x-form-search-trigger',
    hideTrigger1: false,
    width:180,
    hasSearch : false,
    paramName : 'query',

    onTrigger1Click : function(){
        this.setValue('type:');
		locationField.setValue('/');

		this.onTrigger3Click();
    },

    onTrigger2Click : function(){
		this.onTrigger3Click();
    },
	
    onTrigger3Click : function(){
        var v = this.getRawValue();
        if(v.length < 1){
            this.onTrigger1Click();
            return;
        }
	
		this.loader.baseParams.filter = v;
		this.loader.baseParams.path = '/';
		this.tree.root.reload();
    }	
});
