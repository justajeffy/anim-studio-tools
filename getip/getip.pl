#!/usr/bin/perl -w

use strict;

my $ipClass = $ARGV[0];

my $ipDir = "/drd/software/int/getip/";

my $lockTimeout = 0;
while( -e "$ipDir/.lock/locked" ) {
	$lockTimeout++;
	if( $lockTimeout > 10 ) { exit 127 }
	sleep 3;
}
system("touch $ipDir/.lock/locked");

# open the highest number file
open( IP, "<$ipDir/scopes/$ipClass" );
my $highest = <IP>;
close( IP );
chomp $highest;

my @octets = split /\./, $highest;
my $almostlast = $octets[2];
my $last = $octets[3] + 1;

if ( $last > 254 ) {
	$last = 1;
	$almostlast++;
}

my $ip = join(".", ($octets[0], $octets[1], $almostlast, $last ));
print "$ip\n";
open( IP, ">$ipDir/scopes/$ipClass" );
print IP $ip;
close(IP);

unlink "$ipDir/.lock/locked";

