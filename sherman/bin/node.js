var os      = require("os");
var fs      = require("fs");
var exec    = require("child_process").exec;
var dnode   = require("dnode");
var cradle  = require("cradle");
var winston = require("winston");

// Server configuration
var server = require("../conf/server.conf");

var connid;
var logdir;

// Cradle vars
var connection;
var db;

// Logging stuff
var time   = formatDate(new Date());
var logger = new (winston.Logger)({
    transports: [
        new (winston.transports.File)({
            filename: "/var/log/sherman/" + time + ".log",
            timestamp: true
        })
    ],
});

var client = dnode({
   
    restart: function(cb) {

        logger.info("Restarting process");
        
        // Stop this process to reload this script
        process.exit();
    },
    cmd: function(cmd, type, hostid, cb) {
     
        // Should probably restrict what cmd can get executed ...
        // Set the buffer to 250MB, some directories (shd) have thousands of files.
        exec(cmd, { maxBuffer: 1024 * 1024 * 250 }, function(error, stdout, stderr) {
          
            logger.info("");
            logger.info("---");
            logger.info("Command has finished. Processing output.");
            switch (type) {
                case 'files':

                    if ( stdout != '') {

                        // This way only one request can write to the log file.
                        // Need to re-evaluate logging module for multi-writes.
                        fs.open(logdir + "/" + os.hostname() + "_" + connid + ".log", 'a', 0755, function(e, id) {
                            fs.write(id, stdout, null, 'utf8', function() {
                                fs.close(id, function() {

                                    // Now take the data apart and write aggregates to couch
                                    lines = stdout.split("\n");
                                    lines.pop();

                                    var docs = {};
                                    var dids = new Array();
                                    for ( var line = 0; line < lines.length; ++line ) {
                                       
                                        // size, uid, mtime, atime, path
                                        elements = lines[line].split("\t");

                                        // With bad characters in the path the line can be broken up weirdly
                                        // If there are not all 5 elements present, ignore this line
                                        if ( elements.length < 4 )
                                            continue;
 
                                        var path = elements[4].split("/");
                                        path.shift(); // First element is empty

                                        var file = path.pop(); // Last element of path is the file
                                        var type = file.split(".").pop(); // Last element of file is the type
                                        var size = parseInt(elements[0]);
                                        var user = parseInt(elements[1]);

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
                                        docs[id].mtime   = (docs[id].mtime > parseInt(elements[2])) ? docs[id].mtime : parseInt(elements[2]); // Get biggest mtime for folder
                                        docs[id].atime   = (docs[id].atime > parseInt(elements[3])) ? docs[id].mtime : parseInt(elements[3]); // Get biggest atime for folder
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
                                        for (var doc in docs) 
                                            bulk.push(docs[doc]);

                                        // Now save the docs and return
                                        db.save(bulk, function(err, res) {

                                            /* No need to save the dids to file anymore
                                            // instead return the dids and to the comparison on the server
                                            var output = "";
                                            for (var did in dids)
                                                output += dids[did] + "\n";

                                            // Save the document ids/paths to find deleted documents later.
                                            fs.open(logdir + "/" + os.hostname() + "_" + connid + ".dids", 'a', 0755, function(e, id) {
                                                fs.write(id, output, null, 'utf8', function() {
                                                    fs.close(id, function() {

                                                        // All done, callback
                                                        logger.info("Finished writing logs and db entires. Returning to server.");
                                                        return cb('', hostid);
                                                    });
                                                });
                                            });
                                            */

                                            logger.info("Stored documents to Couch.")
                                            return cb('', hostid);
                                        });
                                    });
                                })    
                            })
                        })
                    } else {
                        logger.info("No files output from command. Return to server.");
                        return cb('', hostid);    
                    }

                    break;
                case 'folders':

/*
                    if ( stdout != '' ) {

                        // Get each path
                        var lines = stdout.split("\n");
                        lines.pop(); // Last element is empty
                         
                        var parent   = lines[0].split("/").slice(1, -1);
                        var children = [];
                        for ( var line in lines ) {

                            var child = lines[line].split("/").slice(1).join("/");
                            if ( children.indexOf(child) == -1 )
                                children.push(child);
                        }
                        // Check couch for old documents - remove them if they don't exist on FS anymore
                        db.view('sherman/tree', { startkey: parent.concat([0]), endkey: parent.concat([{}]), group: true, group_level: parent.length + 1 }, function (err, res) {

                            if ( res.length > 0 ) {

                                var del = new Array();
                                res.forEach(function(row) {

                                    // Break up path to compare lengths
                                    var couch = row.path.split("/");
                                    if ( couch.length == parent.length + 1 ) {

                                        if ( children.indexOf(row.path) == -1 ) {

                                            var old      = {};
                                            old._id      = row.path;
                                            old._rev     = row.rev;
                                            old._deleted = true;

                                            del.push(old);
                                        }
                                    }
                                });

                                // Bulk delete
                                if ( del.length > 0 ) {
                           
                                    db.save(del, function(err, res) {
                                    
                                        return cb(stdout, hostid);
                                    });
                                } else {
                                    return cb(stdout, hostid);    
                                }
                            } else {
                                return cb(stdout, hostid);
                            }
                        });

                        //return cb(stdout, hostid);                        
                    } else {
                        return cb(stdout, hostid);
                    }
*/
                    logger.info("Returning directories.");
                                        
                    return cb(stdout, hostid);
                    break;
                default:

                    break;
            }
        });
    }
});

// Connect the client to the server
function connect() {

    client.connect(server.port, server.host, function(remote, conn) {

        // Store the connection ID for reference
        connid = conn.id;

        logger.info("Connecting to " + server.host + " with id: " + connid);

        remote.register(os.hostname(), function(log) {

            // Store the logdir sent from the server
            logdir = log;

            // Set up Cradle for this session
            connection = new (cradle.Connection)('http://' + server.host, 5984, {
                cache: true,
                raw: false
            });
            db = connection.database('sherman');
        });
    });
}

function start() {

    // Wait a few seconds before connecting
    setTimeout(function() { connect(); }, 5000);
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

/*
process.on("uncaughtException", function(err) {

    // There is an issue with the server most likely
    // Try reconnecting in 10 seconds
    logger.info("There was an issue with the connection. Retrying connection.");
    logger.info("Err: " + err);
    setTimeout(function() { connect(); }, 5000);
})
*/

// Start the node
start();
