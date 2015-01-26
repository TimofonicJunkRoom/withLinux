#!/usr/bin/python3.4
# -*- coding: utf=8 -*-

import json
import html
import html.parser
from html.parser import HTMLParser

idc = {}
idc['init', 'tdata'] = []

class MinerHTMLParser (HTMLParser):
	id_cur = 'init'
	def handle_starttag (self, tag, attrs):
		# read all
		for attr, data in attrs:
			if attr == 'data-tweet-id':
				self.id_cur = data
			idc[self.id_cur, str(attr)] = data
			idc[self.id_cur, 'tdata'] = []
	def handle_endtag (self, tag):
		pass
#		print ("\t"*depth + "_etag: ", tag)
	def handle_data (self, data):
#		print ("\t"*(self.depth+1) + str(data))
		idc[self.id_cur, 'tdata'].append(data)
			


def main():
	with open ("pool/linux/1.html", 'r') as f:
		page = f.read()
#print (page)
	parser = MinerHTMLParser()
#	parser = HTMLParser()
	parser.feed (page)
	print (idc)
#	with open ("pool/linux/1.html.json", 'w+') as f:
#		f.write(json.dumps(idc))

#	print (sorted(idc.keys()))

main()
