#!/usr/bin/python3
# Google image downloader

thekeywordlist = ['aireplane+airport']

import os
import requests as req
import re
from bs4 import BeautifulSoup as BSoup
from subprocess import call

# configuration
debug = 1
#baseurl = 'https://{}/search?site=imghp&tbm=isch&source=hp&q='
#baseurl = 'https://www.google.co.jp/search?hl=zh-CN&site=imghp&tbm=isch&source=hp&biw=1265&bih=543'
baseurl = 'https://{0:s}/search?hl=en&site=imghp&tbm=isch&q={1:s}&spell=1&sa=X&ved=0CBgQvwUoAGoVChMIvKu1spv1xgIVgT2UCh22-wzu&dpr=1&biw=1265&bih=543'
uarg = {
	'host' : 'www.google.co.jp',
}

pattern_http = re.compile (r'"http.://.*?"')

# modules
def urlgen (query):
	return baseurl.format(uarg['host'], query)

def download (query):
	_url = urlgen (query)
	page = req.get(_url).text
	print (page)
#a_href = BSoup(page).findAll('img')
	a_href = BSoup(page).findAll('a')
	for item in a_href:
		print (item)
#with open(query, 'r+') as f:
#		f.write (page)
	for item in a_href:
		linkgroup = pattern_http.search (str(item))
		if linkgroup:
			link = linkgroup.group()
			link = re.sub ('"$','', link)
			link = re.sub ('^"','', link)
			print ("I: Download {}".format(link))
#			call ("wget --quiet -P pool/ '{}'".format(link), shell=True)

# main
def main ():
	for keywords in thekeywordlist:
		query = keywords.replace('\xef\xbb\xbf','').replace('\n','').replace('\r','')
		if (debug): print (query, urlgen (keywords))
		download (query)

main()
