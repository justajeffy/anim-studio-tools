#!/usr/bin/perl

use strict;
use CGI qw/:standard/;
use DBI;
use DBD::Pg;

print header, start_html('daNS 1.0');

my $hostname = param('hostname') || (print 'hostname required',br && exit );
my $username = param('username') || (print 'username required',br && exit );
my $os = param('os') || (print 'os required',br && exit );

my $dbh = DBI->connect("dbi:Pg:dbname=mydns;host=sql01.drd.int", "mydns", "Dns1", {AutoCommit => 1, RaiseError => 1});

# first delete the two rows we care about, then insert them.
# we do this so we don't have to select and then update..

my $deleteSql = qq|DELETE FROM rr WHERE name = ?|;
my $sth = $dbh->prepare($deleteSql);
$sth->execute( "$os.$username" );
$sth->execute( "$username" );

# make sure the hostname is properly formatted
if ($hostname !~ /.drd.int$/) { $hostname .= ".drd.int" }
$hostname .= ".";

my $insertSql = qq|INSERT INTO rr (zone, name, type, data, ttl) VALUES (?,?,?,?,?)|;
$sth = $dbh->prepare($insertSql);
$sth->execute( 2, $username, "CNAME", $hostname, 60 );
$sth->execute( 2, "$os.$username", "CNAME", $hostname, 60 );

print "database update successful for $username, $hostname, $os\n", br;

# now update the Snafu / Arsenal database
$hostname =~ s/.drd.int.//;

my $snafuDbh = DBI->connect("dbi:Pg:dbname=blur;host=pgsql.drd.int", "farmer", "brown", {AutoCommit => 1, RaiseError => 1});
my $snafuSql = q|UPDATE host SET fkeyuser = (SELECT keyelement FROM usr WHERE name = ?) WHERE host = ?|;
my $snafuSth = $snafuDbh->prepare($snafuSql);
$snafuSth->execute($username, $hostname);

print "snafu update successful for $username, $hostname, $os\n", br;

