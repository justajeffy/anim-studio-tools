#!/bin/bash
set -x

yum install -y wget
wget http://pkgs.repoforge.org/rpmforge-release/rpmforge-release-0.5.2-2.el6.rf.i686.rpm
rpm -Uvh rpmforge-release-0.5.2-2.el6.rf.i686.rpm 

yum install -y mercurial
hg clone https://code.google.com/p/anim-studio-tools/

yum install -y gcc make gcc-c++

wget http://www.hdfgroup.org/ftp/HDF5/current/src/hdf5-1.8.10.tar.gz
tar -zxvf hdf5-1.8.10.tar.gz
cd hdf5-1.8.10
./configure --prefix=/usr --enable-threadsafe --with-pthread
make install

yum install -y rrdtool-devel
yum install -y python-devel
yum install -y python-genshi
yum install -y python-cherrypy
yum install -y python-simplejson
yum install -y python-setuptools
yum install -y libgfortran
easy_install numpy
easy_install numexpr
easy_install Cython
easy_install tables

cd anim-studio-tools
cd sherman
mkdir input
mkdir output
mkdir rrd
mkdir log
