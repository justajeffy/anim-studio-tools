function convert(bytes) {
	var str    = '';
	var size   = parseInt(bytes);
	var labels = ['KB', 'MB', 'GB', 'TB'];

	for(var j = 0, len = labels.length; j < len; j++){

		if (size > 1024) {
			size /= 1024.0;
			str = size.toFixed(2) + labels[j];
		}
	}

	return str;
}

function constructPath(path) {
	
	var parts = path.split('/');
	var outpt = '|root|';

	parts.shift();
	Ext.each(parts, function(element, index) {
	
		var part = parts.slice(0, index + 1);

		if (part.length > 0)
			outpt += '/' + part.join('/') + '|';
	});

	return outpt;
}
