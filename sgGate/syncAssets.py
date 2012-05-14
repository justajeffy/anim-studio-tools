#!/usr/bin/python2.5

from __future__ import with_statement
from shotgun_api3 import Shotgun
import datetime
import pprint

import pymssql
conn = pymssql.connect(host='sql03', user='USER', password='PASS', database='assystp', as_dict=True)
conn2 = pymssql.connect(host='sql03', user='USER', password='PASS', database='assystp', as_dict=True)


SERVER_PATH = 'http://railgun' # change this to https if your studio uses SSL
SCRIPT_USER = 'sync-facts'
SCRIPT_KEY = '9ea4994e7a51d4550be97c62a80977b134c45399'

sg = Shotgun(SERVER_PATH, SCRIPT_USER, SCRIPT_KEY)

def gather_item_costs(item_id):
    cur2 = conn2.cursor()
    cur2.execute('''
    SELECT
        item_cost_id,
        item_cost_rmk,
        cost_amount
    FROM item_cost
    WHERE item_id = %s
    ''' % item_id)

    cost = float(0.0)
    notes = str()
    row = cur2.fetchone()
    while row:
        pprint.pprint(row)
        if not row['item_cost_id']:
            row = cur2.fetchone()
            continue
        if row['cost_amount']:
            cost = cost + float(row['cost_amount'])
        if row['item_cost_rmk']:
            notes = notes + row['item_cost_rmk'].decode('iso-8859-1') + "\n------------------\n"
        row = cur2.fetchone()
    return [notes, cost]

def transform_asset(row):
    descrip = row['product_n']
    if row['product_n'] and row['product_rmk'] and row['product_rmk'] != 'NULL':
        descrip = row['product_n'] + "\n" + row['product_rmk']
    asset_data = {
        'project': {'type':'Project', 'id':178},
        'code': row['item_sc'].lower(),
        'sg_serial': row['item_serial_no'],
        'sg_acquired_date': datetime.datetime.strptime( str(row['acquired_date']).split(".")[0],"%Y-%m-%d %H:%M:%S"),
        'sg_cost': float(row['item_cost']),
        'sg_cost_center': row['cost_centre_n'],
        'sg_deploy_status': row['item_status_n'],
        'sg_purchase_order_code': row['purch_ord_sc'],
        'description': descrip,
    }
    product_class = row['prod_cls_n']
    if not product_class.find("Unknown") > -1:
        asset_data['sg_product_class'] = product_class

    if row['usr_sc'] and len(row['usr_sc']) > 2:
        username = row['usr_sc'].lower()
        bad_names = ['NULL','STOCK','E3.SUITE2','EC.ROOM','L1NORTH.COMMS','L2WEST.COMMS','L4SOUTH.COMMS','L6.KASHMIR','L1SOUTH.COMMS','UNKNOWN','L6WEST.COMMS','L2SOUTH.COMMS']
        person = sg.find_one('HumanUser', [['login','is',username]], ['id'])
        if person:
            asset_data['sg_assigned_user'] = {'type':'HumanUser', 'id':person['id']}
            sg.update('HumanUser', person['id'], {'name':row['usr_n']})
        elif username and username not in bad_names:
            person = sg.create('HumanUser', {'login':username,'name':row['usr_n']}, return_fields=['id'])
            asset_data['sg_assigned_user'] = {'type':'HumanUser', 'id':person['id']}

    (license_notes, license_costs) = gather_item_costs(row['item_id'])
    asset_data['sg_purchase_notes'] = license_notes
    asset_data['sg_cost'] = asset_data['sg_cost'] + float(license_costs)
    return asset_data


# __main__ 

cur = conn.cursor()
cur.execute('''
SELECT
    item_id,
    item_sc,
    item_serial_no,
    acquired_date,
    item_cost,
    cost_centre_n,
    item_status_n,
    purch_ord_sc,
    usr_sc,
    usr_n,
    p.product_n,
    p.product_rmk,
    p.prod_cls_n
FROM item i
JOIN product p ON i.product_id = p.product_id
WHERE product_n not like 'Unknown %'
ORDER BY item_sc DESC
''')

row = cur.fetchone()
while row:
    if not row['item_sc']:
        row = cur.fetchone()
        continue
    asset_data = transform_asset( row )

    existing_asset = sg.find_one('Asset', [['code','is',asset_data['code']]])

    if existing_asset:
        print "asset %s exists" % ( asset_data['code'] )
        sg.update('Asset', existing_asset['id'], asset_data)
    else:
        print "asset %s creating" % ( asset_data['code'] )
        try:
            sg.create('Asset', asset_data, return_fields=['id'])
            pass
        except Exception:
            pprint.pprint(row)
            pprint.pprint(asset_data)
    row = cur.fetchone()

conn.close()

