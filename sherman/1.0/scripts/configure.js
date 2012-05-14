var dnode = require("../modules/dnode");

// Server config
var conf = require("../conf/master.conf");

/*
var configure = (function() {
    
    var init = function(argv) {
      
        // Parse the args

        
    };

    return {
      init: init  
    };
})();

configure.init(process.argv);
*/

var maxhosts = process.argv[2];
console.log("Setting maxhosts to " + maxhosts);

// Connect the client to the server
function connect() {

    dnode.connect(conf.port, conf.host, function(remote, conn) {

        remote.setMaxHosts(maxhosts, function(msg) {

			console.log(msg);

			process.exit();
        });
    });
}

connect();
