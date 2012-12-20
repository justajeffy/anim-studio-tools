#!/bin/bash
set -x

wget http://pkgs.repoforge.org/rpmforge-release/rpmforge-release-0.5.2-2.el6.rf.i686.rpm
yum install mercurial
hg clone https://code.google.com/p/anim-studio-tools/
yum install rrdtool-devel
rpm -Uvh rpmforge-release-0.5.2-2.el6.rf.i686.rpm 
yum install hdf5-devel
yum install python-genshi
yum install python-cherrypy
yum install python-simplejson
easy_install numpy
easy_install numexpr
easy_install Cython
easy_install tables

cd anim-studio-tools
cd sherman
mkdir input
mkdir output
mkdir rrd
