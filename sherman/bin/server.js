var fs      = require("fs");
var path    = require("path");
var dnode   = require("dnode");
var winston = require("winston");

// Load config 
var server = require("../conf/server.conf");

// Collector configuration
var collector = require("../conf/collector.conf");

// Get the configured paths
var paths = require("../conf/paths.conf");

// General vars
var time       = formatDate(new Date());
var logdir     = "/farm/logs/sherman/input/" + time;

// Dispatcher data stores
var pathlist    = paths.list;
var busy_hosts  = { 'count': 0 };
var ready_hosts = new Array();

// Set a maxium of hosts to work with
// Don't want to overload the storage
var maxhosts = collector.maxhosts;

// Host number counters
var usable     = 0;
var registered = 0;

// Logging stuff
var logger     = new (winston.Logger)({
    transports: [
        new (winston.transports.File)({ 
            filename: "/var/log/sherman/" + time + ".log",
            timestamp: true
        })
    ],
});

// Commands to run
var files   = "find \"%PATH%\" -mindepth 1 -maxdepth 2 -type f -printf \"%s\t%U\t%T@\t%A@\t%p\n\" -o -wholename '.*snapshot*' -prune";
var folders = "find \"%PATH%\" -mindepth 2 -maxdepth 2 -type d -printf \"SHR\t%p\n\" | grep -v /.snapshot/"; // Annoying to exclude via grep ...

function prepareDispatch() {

    // Exit when nothing else to do
    if ( pathlist.length == 0 && busy_hosts.count == 0 ) {
        console.log("Done!");
        process.exit();
    }

    console.log("");
    console.log("Starting Loop ...");
    console.log(pathlist.length + " paths to be dispatched.");
    console.log(ready_hosts.length + " hosts are ready.");
    console.log(busy_hosts.count + " hosts are still busy.");
    console.log(usable + "/" + registered + " hosts getting used.");
	console.log("Max hosts: " + maxhosts);

	// Adjust ready_hosts like according to maxhosts
	// A good way to throttle number of nodes
	if ( usable < maxhosts && registered > usable )

    // Check the busy_hosts to see which paths they are still working on
    if ( busy_hosts.count > 0 && busy_hosts.count < 20 ) {
        
        console.log("");
        console.log("Hosts still busy: ");
        for ( var host in busy_hosts ) {
           
            if ( busy_hosts[host].name ) 
                console.log(busy_hosts[host].name + " is still busy with " + busy_hosts[host].path);
        }
    }

    // Start the next loop
    dispatch();
}

function restart() {
    
    for (var node in ready_hosts) {
        
        var host = ready_hosts[node];

        console.log("Restarting " + host.name);
        host.host.restart();
    }

    // After restart quit
    process.exit();
}

function dispatch() {
 
    while (pathlist.length > 0) {
       
        // Get the first host in queue
        host = ready_hosts.shift();

        if (host) {

            // Get the first path in the list
            path = pathlist.shift();

            //console.log("Sending Path: " + path + " to " + host.name);

            // Put the host on the busy list
            busy_hosts[host.id] = host;
            busy_hosts[host.id].path = path;
            busy_hosts.count++;

            //console.log("Using host " + host.name + " for path " + path);

            // Find files
            host.host.cmd(files.replace('%PATH%', path), 'files', host.id, function(result, hostid) {

                // Put this host back on the queue
                ready_hosts.push(busy_hosts[hostid]);

                // Delete from busy queue
                delete busy_hosts[hostid];
                busy_hosts.count--;
            });

            // Find folders
            host.host.cmd(folders.replace('%PATH%', path), 'folders', host.id, function(result, hostid) {

                if (result != '') {
                  
                    remotePaths = result.split("\n");
                    remotePaths.pop(); // There is an empty element at the end of the array

                    // Combine existing paths array with new/remote set of paths
                    for (var p in remotePaths) {

                        // Special characters can screw up the output
                        // To avoid confusion check for the prepended SHR flag
                        // which will only appear on non-broken lines
                        cleanPath = remotePaths[p].split("\t");
                        if ( cleanPath[0] == "SHR" )
                            pathlist.push(cleanPath[1]);
                    }

                    // Put this host back on the queue
                    // NOTE: atm this callback only gets called from the directory find
                    // so there is a chance that the host is still running the file find cmd
                    // meaning multiple find can runn at the same time on one host ...
                    //ready_hosts.push(busy_hosts[hostid]);

                    // Delete from busy queue
                    //delete busy_hosts[hostid];
                }
            });
        } else {
            
            break;
        }
    }

    setTimeout(function() {
        prepareDispatch();
    }, 100);
}

function return_host(host) {

	ready_hosts.push(busy_hosts[hostid]);
}

// Format the date to 'yyyy-mm-dd-hh-mm-ss'
function formatDate(d) {
    
    function pad(n) { return n < 10 ? '0' + n : n }

    return d.getUTCFullYear() + '-'
        + pad(d.getUTCMonth() + 1) + '-'
        + pad(d.getUTCDate()) + '-'
        + pad(d.getUTCHours()) + '-'
        + pad(d.getUTCMinutes()) + '-'
        + pad(d.getUTCSeconds())
}

function main() {

    // Create collection log directory if it does not exist
    if ( !path.existsSync(logdir) )
        fs.mkdirSync(logdir, 0755);

    // Initialize dnode object
    dnode(function(client, conn) {
    
        this.register = function(hostname, cb) {

            // Create the host object from remote client
            host = {};
            host.id   = conn.id;
            host.host = client;
            host.name = hostname;
            host.path = "";

            // Add host to the ready queue
            // Limit to maxhosts
            if ( usable < maxhosts ) {
                ready_hosts.push(host);
                usable++;
            }

            // Done registering host
            registered++;
            console.log("Registered " + hostname + " with id: " + host.id);

            // Return log dir for this session
            return cb(logdir);
        };

		this.configure = function(conf, cb) {

			// Change collector settings based on what's passed in
			maxhosts = conf.maxhosts;

			console.log("Max host has been changed");

			return cb("Success!");
		};

    }).listen(server.port, server.host);

    // Wait a bit to have at least a few nodes connect for dispatch
    console.log("Waiting 10 seconds for nodes to join.");
    
    // Start dispatching
    setTimeout(function() {
        console.log("Let's Go!");
        prepareDispatch();
    }, 10000);

    // Restart all host to pick up changes
    //setTimeout(function() {
    //    restart();
    //}, 15000);
}

// Start Server
console.log("Starting Server.");
main();
