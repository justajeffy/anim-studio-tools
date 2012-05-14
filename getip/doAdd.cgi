#!/usr/bin/perl -w
use strict;
use CGI qw/:standard/;
use Net::MAC;

print header, start_html('autoAdd 1.0');

my $hostname = param('hostname') || (print 'hostname required',br && exit );
my $mac = param('mac') || ( print 'mac required',br && exit );
my $macObj = Net::MAC->new('mac' => $mac); 
my $scope = param('scope') || ( print 'scope required',br && exit );

if( param('hostname') && param('mac') && param('scope') ) {
	chdir('/drd/software/int/getip');

	# first get next available IP
	my $ipCmd = "./getip.pl $scope";
	print "running $ipCmd",br;
	my $ip = `$ipCmd`;

	print "got IP: $ip",br;

	open(LOG,">>log/register.log");
	print LOG "$hostname\t$mac\t$scope\t$ip\n";
	close(LOG);

	my $registerCmd = "./registerWithAD.pl $hostname $mac $ip";
	print $registerCmd, br;

	my $results = `$registerCmd`;
	print $results, br;
}

