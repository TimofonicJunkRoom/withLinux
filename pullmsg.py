#!/usr/bin/python3.4

# cdluminate@163.com

# simply pull messages from twitter
# using https://twitter.com/serach?q=

import os
import requests
import re
import urllib

debug = 1

def do_search (keyword, debug):
	"""
	send the request to server
	return the page.text server returned here
	"""
	if debug: print ("searching for {}".format(keyword))

	url = "https://twitter.com/search?q=" + keyword
	if debug: print ("url : {}".format(url))

	page = requests.get (url)
	#print (page.text)
	return page.text

def main ():
# check pool/archive dir
	if not os.path.exists ("pool"):
		os.makedirs ("pool")
	if not os.path.exists ("pool/archive"):
		os.makedirs ("archive")
# check list.lst	
	if not os.path.exists ("list.lst"):
		print ("E : list.lst missing")
		exit (1)
# open parsed list
	f1lst = open ("pool/f1.lst")
	f2lst = open ("pool/f2.lst")
# readline
	count = 0
	if debug: print ("Searching field 1")
	for line in f1lst.readlines():
		if debug: print ("{} : ".format(count) + line)
		page = do_search (line.replace(' ', '+'), debug)
		_f = open ("pool/archive/{}.f1".format(count), "w+")
		_f.write (page)
		_f.close ()
		count = count + 1

	count = 0
	if debug: print ("Seaching field 2")
	for line in f2lst.readlines():
		if debug: print ("{} : ".format(count) + line)
		page = do_search (line.replace(' ', '+'), debug)
		_f = open ("pool/archive/{}.f2".format(count), "w+")
		_f.write (page)
		_f.close()
		count = count + 1

	f1lst.close()
	f2lst.close()
	
	exit()

main ()
