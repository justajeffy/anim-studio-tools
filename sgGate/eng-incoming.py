import imaplib
import parse
from shotgun_api3 import Shotgun

SERVER_PATH = 'http://railgun' # change this to https if your studio uses SSL
SCRIPT_USER = 'mail-gateway'
SCRIPT_KEY = '987fd62624610c1580c070441f1ae208653541f9'

sg = Shotgun(SERVER_PATH, SCRIPT_USER, SCRIPT_KEY)

M = imaplib.IMAP4_SSL('imap.gmail.com', 993)
M.login('EMAIL@SERVER.COM', 'PASSWORD')
M.select("eng-support") # mailbox/tag name
typ, data = M.search(None, 'ALL')
for num in data[0].split():
    typ, msgdata = M.fetch(num, '(RFC822)')
    #print 'Message %s\n%s\n' % (num, data[0][1])
    msg_dict = parse.parse(msgdata[0][1])
    print "message from %s" % msg_dict["from"]

    people = sg.find("HumanUser",[['email','is',msg_dict["from"]]],['id','name'])
    if len(people) < 1:
        print "couldn't find user"
    else:
        # if we find a valid user, create a ticket for them
        user = people[0]
        ticket_data = {
            'created_by': {'type':'HumanUser','id':user['id']},
            'addressings_to': [{'type':'Group','id':5}],
            'title': msg_dict['subject'],
            'description': msg_dict['body'],
            'project': {'type':'Project', 'id':178},
        }
        sg.create('Ticket', ticket_data, return_fields=['id'])

        # if we made it this far the e-mail was processed, now delete it
        M.store(num, '+FLAGS', '\\Deleted')

M.close()
M.logout()

