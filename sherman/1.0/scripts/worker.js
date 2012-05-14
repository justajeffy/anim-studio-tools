var os = require("os"),
    fs = require("fs"),
  exec = require("child_process").exec,
 dnode = require("../modules/dnode"),
cradle = require("../modules/cradle"),
master = require("../conf/master.conf");

var restart = function() {

	// Trying to re-connect
	setTimeout(function() { worker.init(); }, 5000);
}

/*
process.on("uncaughtException", function(err) {

    // There is an issue with the server most likely
    // Try reconnecting in 10 seconds
    setTimeout(function() { worker.init(); }, 5000);
})
*/

var worker = (function() {

	var hostname = os.hostname();
	var id, db, logdir;

	var api = {

		md5: function(path, hostid, cb) {

			var cmd = "/drd/software/int/sys/sherman/tmp/md5.sh '" + path + "'";
			exec(cmd, { maxBuffer: 1024 * 1024 * 250 }, function(error, stdout, stderr) {
	
				if ( stdout != '' ) {
                    
					// Write to log file
                    fs.open(logdir + "/" + hostname + "_" + id + ".log", 'a', 0755, function(e, fd) {
                        fs.write(fd, stdout, null, 'utf8', function() {
                            fs.close(fd, function() {

								return cb('', hostid);
							});
						});
					});
                } else {
                    return cb('', hostid);
                }
			});
		},

		find: function(path, mindepth, maxdepth, exclude, couch, hostid, cb) {
			//console.log("Finding files/folders for " + path);

			// Put together the find command
			var cmd = "/usr/bin/find '" + path + "'" +
                      " -mindepth " + mindepth + 
                      " -maxdepth " + maxdepth +
                      " -type f" +
                      " -printf \"%s\t%U\t%T@\t%A@\t%p\n\"" + 
					  //" -printf \"%s\t%p\n\"" +
                      " -o -wholename '.*snapshot*' -prune" +
					  " -o -wholename '/drd/jobs/hf2/tank*' -prune";
			exec(cmd, { maxBuffer: 1024 * 1024 * 250 }, function(error, stdout, stderr) {
	
				if ( stdout != '' ) {
                    
					// Write to log file
                    fs.open(logdir + "/" + hostname + "_" + id + ".log", 'a', 0755, function(e, fd) {
                        fs.write(fd, stdout, null, 'utf8', function() {
                            fs.close(fd, function() {

								if ( !couch ) {

									// Don't store anything in couchdb
									return cb('', hostid);
								} else {
    								// Now take the data apart and write aggregates to couch
    								lines = stdout.split("\n").slice(0, -1);

    								var docs = {};
                                    var dids = new Array();
                                    for ( var line = 0; line < lines.length; ++line ) {
                                       
                                        // size, uid, mtime, atime, path
                                        elements = lines[line].split("\t");

                                        // With bad characters in the path the line can be broken up weirdly
                                        // If there are not all 5 elements present, ignore this line
                                        if ( elements.length < 4 )
                                            continue;

                                        var path = elements[4].split("/").slice(1);

                                        var file  = path.pop(); // Last element of path is the file
                                        var type  = file.split(".").pop(); // Last element of file is the type
                                        var size  = parseInt(elements[0]);
                                        var user  = parseInt(elements[1]);
    									var mtime = parseInt(elements[2]);
    									var atime = parseInt(elements[3]);

                                        // Join path back together as id for document
                                        var id = path.join("/");

                                        // Make document id the path
                                        try {
                                            docs[id]["_id"] = id;
                                        } catch(err) {

                                            // Create a fresh document object
                                            docs[id] = {
                                                "_id"      : id,
                                                "path"     : id,
                                                "size"     : 0,
                                                "mtime"    : 0,
                                                "atime"    : 0,
                                                "nfiles"   : 0,
                                                "users"    : {},
                                                "types"    : {},
                                                "user_type": {}
                                            };

                                            // Fill the doc ids array for later bulk get
                                            dids.push(id);
                                        }

                                        docs[id].size   += size; // Add up file sizes
                                        docs[id].mtime   = (docs[id].mtime > mtime) ? docs[id].mtime : mtime; // Get biggest mtime for folder
                                        docs[id].atime   = (docs[id].atime > atime) ? docs[id].mtime : atime; // Get biggest atime for folder
                                        docs[id].nfiles += 1;

                                        // Create the file type store
                                        try {
                                            docs[id].types[type][0] += 1;    // number of files
                                            docs[id].types[type][1] += size; // size of file
                                        } catch(err) {
                                            docs[id].types[type] = [1, size];
                                        }

                                        // Create the user store
                                        try {
                                            docs[id].users[user][0] += 1;    // number of files
                                            docs[id].users[user][1] += size; // size of file
                                        } catch(err) {
                                            docs[id].users[user] = [1, size];
                                        }

                                        // Create the user-type store
                                        try {
                                            docs[id].user_type[user][type][0] += 1;    // number of files
                                            docs[id].user_type[user][type][1] += size; // size of files
                                        } catch(err) {

                                            try {
                                                docs[id].user_type[user][type] = [1, size];
                                            } catch(err) {
                                                docs[id].user_type[user] = {};
                                                docs[id].user_type[user][type] = [1, size];
                                            }
                                        }
                                    }

                                    // Create correct structure for the bulk insert/update
                                    var bulk = new Array();
                                    db.get(dids, function(err, prevdocs) {

                                        for (var prevdoc in prevdocs) {
                                           
                                            try {
                                                var rev   = prevdocs[prevdoc].value.rev;
                                                var docid = prevdocs[prevdoc].id;

                                                // Update the doc to know the revision
                                                docs[docid]["_rev"] = rev;

                                                // If nothing has changed since last revision no need to save it
                                                // This will save on couchdb reduce time.
                                                if ( docs[docid].size   == prevdocs[prevdoc].doc.size  &&
                                                     docs[docid].mtime  == prevdocs[prevdoc].doc.mtime &&
                                                     docs[docid].atime  == prevdocs[prevdoc].doc.atime &&
                                                     docs[docid].nfiles == prevdocs[prevdoc].doc.nfiles ) {
                                               
                                                    delete docs[docid];
                                                }
                                            } catch(err) {
                                                // The document does not have a previous revision
                                            }
                                        }

                                        // Create correct structure for the bulk insert/update
                                        for (var doc in docs) {
                                            bulk.push(docs[doc]);
                                        }

                                        // Now save the docs and return
                                        db.save(bulk, function(err, res) {
                                            return cb('', hostid);
                                        });
                                    });
								}
							});
						});
					});
				} else {
					return cb('', hostid);
				}
			});
		},

		finddir: function(path, mindepth, maxdepth, exclude, hostid, cb) {

			// Put together the find command
			var cmd = "/usr/bin/find '" + path + "'" + 
                      " -mindepth " + mindepth + 
                      " -maxdepth " + maxdepth +
                      " -type d"  +
                      " -printf \"SHR\t%p\n\" | grep -v /.snapshot/";
			exec(cmd, { maxBuffer: 1024 * 1024 * 250 }, function(error, stdout, stderr) {

				// Return all paths to master
				return cb(stdout, hostid);
			});
		}
	}

	var connect = function() {

		dnode(api).connect(master.port, master.host, function(remote, conn) {

            remote.register(hostname, function(connid, log, msg) {

				// Store the registered connection id
				id = connid;

				// Store the log directory
				logdir = log;
				
				//console.log("Msg: " + msg);

                // Set up Cradle for this session
                var connection = new (cradle.Connection)('http://' + master.host, 5984, {
                    cache: true,
                    raw: false
                });
                db = connection.database('sherman');
            });

    		// Send heartbeat
    		var pulse = setInterval(function() {
    			remote.pulse(hostname, id, "/", function(msg) {
    				//console.log("Msg: " + msg + " " + id);
    			});
    		}, 60000);

			conn.on('end', function() {

				// Stop pulsing
				clearInterval(pulse);

				// Restart worker
				restart();
			});
		});
	};

	var init = function() {
		connect();
	};

	return {
		init: init
	};
})();

worker.init();
