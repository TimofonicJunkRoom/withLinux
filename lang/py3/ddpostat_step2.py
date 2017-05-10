#!/usr/bin/python3
'''
DDPO Statistics
'''
import simplejson as json
import urllib3 as ul3

#UDDURL = "https://udd.debian.org/dmd/?format=json&email1="
UDDURL = {}
pm = ul3.PoolManager()

def mailaddr (login = ""):
	return (login + "@debian.org").strip()

def geturl (login):
	return ("https://udd.debian.org/dmd/?email1=" + mailaddr(login) +
			"&email2=&email3=&packages=&ignpackages=&format=json#versions")

with open("step1.out") as f:
	for line in f:
		login = line.strip()
		login = "aron" # for debug
		print ("I: processing login [ " + login + " ]" )
		url = geturl(login)
		print ("   " + url)
		pass
		response = pm.request("GET", url)
		result = str( response.data ).strip()
		'''print (json.dumps(result))'''
		reslist = json.loads(str(response.data).strip())
		'''type (dresult)'''
		print (reslist)
		break
#print (result)
