var path = require("path");
      fs = require("fs");
     net = require("net");
   dnode = require("../modules/dnode"),
  master = require("../conf/master.conf");

// Get command line arguments
var config, timestamp;
if ( process.argv.length > 2 ) {
	config    = process.argv[2]
	timestamp = process.argv[3];
}

// Host constructor function
function Host(id, worker, name, pulse) {
	this.id     = id;
	this.worker = worker;
	this.name   = name;
	this.pulse  = pulse;
	this.path   = "";
	this.jobs   = 0;
}

// Format the date to 'yyyy-mm-dd-hh-mm-ss'
function formatDate(d) {
    
    function pad(n) { return n < 10 ? '0' + n : n }

    return d.getUTCFullYear() + '-'
        + pad(d.getUTCMonth() + 1) + '-'
        + pad(d.getUTCDate()) + '-'
        + pad(d.getUTCHours() + 11) + '-'
        + pad(d.getUTCMinutes()) + '-'
        + pad(d.getUTCSeconds())
};

var sherman = (function() {

	var paths    = master.paths,
	    maxhosts = master.maxhosts,
	    maxpulse = master.maxpulse;

	var hosts         = {},
	    busy          = { count: 0 },
	    available     = new Array(),
	    broken_paths  = { count: 0 },
		registered    = 0,
		deleted_hosts = new Array();

	var timestamp = formatDate(new Date()),
	       logdir = "/farm/logs/sherman/input/" + timestamp,
          stopped = false;

    var ping = function(host) {
      
        net.createConnect(master.port, host.name).on("connect", function(e) {
            return true;
        }).on("error", function(e) {
            return false;
        });
    };

    var getHost = function() {
    
        while (available.length > 0) {
        
            host = available.shift();

            // Ping the host
            if ( ping(host) ) {
                return host;
            } else {
                continue;
            }
        }
    };

	var dispatch = function() {

		while (paths.length > 0) {

			if ( busy.count >= maxhosts )
				break;

            // Get a healthy host
            host = available.shift();

			if ( host ) {

                // Get a randcom path from the list
                //path = paths.shift();
				path = paths.splice(parseInt(Math.random() * paths.length), 1);

                // Put the host on the busy list
				host.path     = path;
				host.jobs    += 2;
                busy[host.id] = host;
                busy.count++;

/*
				// Run md5sum against files
 				host.worker.md5(path, host.id, function(result, hostid) {

 					// That's all
 					busy[hostid].jobs -= 2;
 				});
*/

                // Find files
                host.worker.find(path, 1, 2, "", false, host.id, function(result, hostid) {

					busy[hostid].jobs--;
                });

				// Find subdirectories
				host.worker.finddir(path, 2, 2, "", host.id, function(result, hostid) {

                    if (result != '') {
                      
                        var remotePaths = result.split("\n").slice(0, -1); // There is an empty element at the end of the array

                        // Combine existing paths array with new/remote set of paths
                        for (var p in remotePaths) {

                            // Special characters can screw up the output
                            // To avoid confusion check for the prepended SHR flag
                            // which will only appear on non-broken lines
                            cleanPath = remotePaths[p].split("\t");
                            if ( cleanPath[0] == "SHR" ) {

                                // Balance the distibution by pre/appending the paths
                                // This should avoid targetted hits on specific volumes/filers
                                // for longer periods of time.
                                var num = parseInt(Math.random()*2);
                                if ( num > 0 )
                                    paths.push(cleanPath[1]);
                                else
                                    paths.unshift(cleanPath[1]);
							}
                        }
                    }

					busy[hostid].jobs--;			
				});

			} else {
				break;
			}
		}

		// Quick breather and prepare next loop
        setTimeout(function() {
            prepare();
        }, 500);
	};

	var check_hosts = function() {

		// Make sure there is only one time to compare against
		var now = getTime();

		// Find hosts in the busy list that don't have jobs anymore
		if ( busy.count > 0 ) {
		
			for ( var hostid in busy ) {

				// If this host is not doing anything anymore
				// or has not pulsed in some time
				if ( busy[hostid].jobs == 0 || busy[hostid].pulse + maxpulse < now ) {
		
					// Put it back in the avaiable queue
					available.push(busy[hostid]);

                    // Delete from busy queue
                    delete busy[hostid];
                    busy.count--;
				}
			}
		}

		// Check health of available hosts
		// Any host with older pulse than maxpulse
		// needs to be looked at
		var to_be_deleted   = new Array();
		var available_count = available.length;
		for ( var i = 0; i < available_count; ++i ) {

			var host = available[i];
			if ( host.pulse + maxpulse < now ) {

				// Something's wrong
				console.log("Pulse time: " + host.pulse);
				console.log("Max Pulse time: " + maxpulse);
				console.log("Now: " + now);
				if ( host.path != '' ) {
			
					// Remember this path, maybe it's 
					// the reason why the worker broke
					if ( broken_paths[host.path] ) {
						broken_paths[host.path]++;
					} else {
						broken_paths[host.path] = 1;
						broken_paths.count++;
					}

					// If this path has been part of a broken host
					// a few time then there is no point in retrying ..
					if ( broken_paths[host.path] < 3 ) {
						paths.push(host.path);
					}
				}

				// Delete the host, it has probably already re-registered
				console.log("Deleting host " + host.name);
				to_be_deleted.push(i);
				deleted_hosts.push(host.name);
				delete hosts[host.id];
			}
		}

		// Delete any hosts from available queue
		var to_be_deleted_count = to_be_deleted.length;
		if ( to_be_deleted.length > 0 ) {
			for ( var i = 0; i < to_be_deleted_count; ++i ) {
				available.splice(to_be_deleted[i], 1);
                registered--;
			}
		}
	};

	var prepare = function() {

		check_hosts();

        // Exit when nothing else to do
        if ( paths.length == 0 && busy.count == 0 && !stopped ) {
            console.log("Done!");

            stopped = true;

			// Write out the receipt
            fs.open("/var/log/sherman/collector-receipt", 'w', 0777, function(e, fd) {
                fs.write(fd, timestamp, null, 'utf8', function() {
                    fs.close(fd, function() {

						// All done for this run
						process.exit();
					});
				});
			});
        } else {
            stopped = false;
        }

        console.log("");
        console.log("Starting Loop ...");
        console.log("           Paths: " + paths.length);
		console.log("    Broken Paths: " + JSON.stringify(broken_paths));
		console.log("Registered Hosts: " + registered);
        console.log(" Available Hosts: " + available.length);
        console.log("Still Busy Hosts: " + busy.count);
		console.log("       Max Hosts: " + maxhosts);
		console.log("   Deleted Hosts: " + deleted_hosts);

        // Check the busy_hosts to see which paths they are still working on
        if ( busy.count > 0 && busy.count < 20 ) {

            console.log("");
            console.log("Hosts still busy: ");
            for ( var host in busy ) {

                if ( busy[host].name )
                    console.log(busy[host].name + " is still busy with " + busy[host].path);
            }
        }

		dispatch();
	};

	var startup = function() {

		dnode(function(worker, conn) {
    
        	this.register = function(hostname, cb) {

    			// Create host object
    			var host = new Host(conn.id, worker, hostname, getTime());

				// Keep a host list
				hosts[conn.id] = host;

				// Add host to available list
				available.push(host);

				// Number of registered hosts
				registered++;

				console.log("Registered " + hostname + " with id " + conn.id);

				return cb(conn.id, logdir, "Thanks for registering!");
        	};

			this.pulse = function(hostname, connid, path, cb) {

    			// Update host pulse
				hosts[connid].pulse = getTime();
    			//console.log("Host " + hostname + " with connId " + connid + " has pulsed.");
				
    			return cb("Awesome!");
    		};

            this.setMaxHosts = function(maxHosts, cb) {
                maxhosts = maxHosts;

                return cb("MaxHosts has been set to " + maxHosts);
            };
    	}).listen(master.port, master.host);
	};

	var createLogDir = function() {

		// Create collection log directory if it does not exist
        if ( !path.existsSync(logdir) )
            fs.mkdirSync(logdir, 0777);
	};

	var getTime = function() {

		// getTime() returns in ms ..
		return Math.round(new Date().getTime()/1000.0);
	}

	var init = function() {
		
		console.log("Initializing.");

		createLogDir();

		startup();

		// Wait for hosts to register
		setTimeout( function() { prepare() }, 10000);
	};

	return {
		init: init
	};
})();

sherman.init();
