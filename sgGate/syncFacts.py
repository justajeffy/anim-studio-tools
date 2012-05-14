#!/usr/bin/python2.5

from shotgun_api3 import Shotgun
from socket import gethostname
import subprocess

SERVER_PATH = 'http://railgun' # change this to https if your studio uses SSL
SCRIPT_USER = 'sync-facts'
SCRIPT_KEY = '9ea4994e7a51d4550be97c62a80977b134c45399'

sg = Shotgun(SERVER_PATH, SCRIPT_USER, SCRIPT_KEY)

facts_to_sync = ['nvidiacard',
                 'nvidiadriver',
                 'nic',
                 'serialnumber',
                 'puppetversion',
                 'productname',
                 'processorcount',
                 'physicalprocessorcount',
                 'operatingsystem',
                 'operatingsystemrelease',
                 'memorysize',
                 'kernelrelease',
                 'macaddress',
                 'netspeed',
                ]

def get_fact_data():
    ret = {}
    raw_facts = subprocess.Popen(["facter"], stdout=subprocess.PIPE).communicate()[0]
    for line in raw_facts.split("\n"):
        try: line.index("=>")
        except: continue
        #print "facter: %s" % line
        (key, value) = line.split("=>")
        #print "fkey: %s fvalue: %s" % ( key, value )
        ret[key.strip()] = value.strip()
    return ret

hostname = gethostname().split('.', 1)[0]

asset = sg.find_one("Asset",[['code','is',hostname]],['id'])
if not asset:
    print "couldn't find asset"
else:
    # if we find a valid asset, sync the facts for it
    fact_data = get_fact_data()

    for fact_name,fact_value in fact_data.items():
        if not fact_name in facts_to_sync: continue

        fact_data = {
            'code': fact_name,
            'sg_asset': {'type':'Asset', 'id':asset['id']},
            'description': fact_value,
            'project': {'type':'Project', 'id':178},
        }
        existing_fact = sg.find_one('CustomEntity01', [['code','is',fact_name],['sg_asset','is',{'type':'Asset','id': asset['id']}]],['id'])
        if existing_fact:
            print "asset %s has existing fact %s, updating to %s" % ( hostname, fact_name, fact_value )
            sg.update('CustomEntity01', existing_fact['id'], fact_data)
        else:
            print "asset %s creating fact %s : %s" % ( hostname, fact_name, fact_value )
            sg.create('CustomEntity01', fact_data, return_fields=['id'])

