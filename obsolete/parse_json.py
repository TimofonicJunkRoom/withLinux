#!/usr/bin/python3.4

import sys
import json
import os

debug = 1

def main (argv):
	if debug: print (sys.argv)
	if (sys.argv[1] == ""):
		print ("missing argv[1]")
		exit(1)
	f = open (sys.argv[1], "r")
	page = f.read ()
	f.close()
	jpage = json.loads(page)
	f = open (sys.argv[1] + ".p", "w+")
	f.write (str(jpage))
	f.close ()

	return 0

argv = {""}
main(argv)
