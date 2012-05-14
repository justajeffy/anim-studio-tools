#!/bin/env python2.5

import os
import re
import sys
import datetime

from farmsubmit import *

###

TIMESTAMP = sys.argv[1]

def gather():

    # Child job to generate db file
    s = {}
    s["jobType"]         = "Batch"
    s["job"]             = "Generate Sherman DB"
    s["cmd"]             = "LD_LIBRARY_PATH=/drd/software/packages/hdf5/1.8.5/lin64/ext/lib python2.5 /drd/software/int/sys/sherman/bin/generate.py %s 2> /dev/null" % TIMESTAMP
    s["packetsize"]      = "1"
    s["assignmentSlots"] = "1"
    s["frameList"]       = "1-1"
    s["priority"]        = 20
    s["maxErrors"]       = "4"
    s["minMemory"]       = "12582912"
    s["maxMemory"]       = "16777216"
    s["maxTaskTime"]     = "28800"
    s["maxQuietTime"]    = "30000"
    s["user"]            = "jay.munzner"

    job = FarmJob(s)
    job.submit()

if __name__ == '__main__':

    # Start the gather
    gather()
