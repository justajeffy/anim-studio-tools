#!/usr/bin/perl -w
use strict;

my $hostname 	= lc($ARGV[0]);
($hostname) = split /\./, $hostname;

my $mac 			= lc($ARGV[1]);
my $ip 				= $ARGV[2];

$mac =~ s/://g;
$mac =~ s/-//g;

my @octets = split /\./, $ip;
my $almostlast = $octets[2];
my $last = $octets[3];

my $scope = "10.19.32.0";
my $dhcpServer = "kmmdc1";
if( $almostlast < 32 ) {
	# this thing must be in the Yurongee Plataeu
	$scope = "10.19.0.0";
	$dhcpServer = "auyurdc02";
}

my $dhcp = qq|netsh dhcp server \\\\$dhcpServer scope $scope add reservedip 10.19.$almostlast.$last $mac $hostname autoAdd1.0|;
my $dhcpOption = qq|netsh dhcp server \\\\$dhcpServer scope $scope set reservedoptionvalue 10.19.$almostlast.$last 012 string $hostname|;

my $forwardDns = qq|dnscmd kmmdc1 /recordadd drd.int $hostname A 10.19.$almostlast.$last|;
my $reverseDns = qq|dnscmd kmmdc1 /recordadd 19.10.in-addr.arpa $last.$almostlast PTR $hostname.drd.int.|;

my $dhcpCmd = qq|./winexe -U DRD.INT/adminregister\%Register1 //$dhcpServer '$dhcp'|;
my $results = `$dhcpCmd`;
print "$dhcpCmd <br>\n";
print "$results <br>\n";

my $dhcpOptionCmd = qq|./winexe -U DRD.INT/adminregister\%Register1 //$dhcpServer '$dhcpOption'|;
$results = `$dhcpOptionCmd`;
print "$dhcpOptionCmd <br>\n";
print "$results <br>\n";

my $forwardDnsCmd = qq|./winexe -U DRD.INT/adminregister\%Register1 //kmmdc1 '$forwardDns'|;
$results = `$forwardDnsCmd`;
print "$forwardDnsCmd <br>\n";
print "$results <br>\n";

my $reverseDnsCmd = qq|./winexe -U DRD.INT/adminregister\%Register1 //kmmdc1 '$reverseDns'|;
$results = `$reverseDnsCmd`;
print "$reverseDnsCmd <br>\n";
print "$results <br>\n";

#print join( "\n", ( $dhcpCmd, $dhcpOptionCmd, $forwardDnsCmd, $reverseDnsCmd ) );

