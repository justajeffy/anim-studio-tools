#!/bin/bash

RRD=/opt/sherman/rrd/$1

/usr/bin/rrdtool xport --start now-120h --end now DEF:size=$RRD:size:LAST XPORT:size | sed -n -e 's/.*<v>\(.*\)<\/v>.*/\1/p'
