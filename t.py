#!/usr/bin/python3.4
# translate argv with bing
# 2015 <cdluminate@163.com>

import sys
import requests
from html.parser import HTMLParser

class DictParser (HTMLParser):
	depth = 0
	def handle_starttag (self, tag, attrs):
		#self.depth = self.depth + 1
		#print ("  "*self.depth + tag + "{")
		for attr, data in attrs:
			if attr == "content":
				if data.find("必应词典") == 0:
					print (data.replace("，", "，\n").replace(".", ".\n").replace("；", "；\n").replace("：", "：\n"))
			#print (attr, data)
	def handle_endtag (self, tag):
		self.depth = self.depth - 1
		#print ("  "*self.depth + "}")
	def handle_data (self, data):
		pass
		#print (data)

# configure
debug   = 1
baseurl = "https://www.bing.com/dict/search?q="
quiry   = baseurl
parser  = DictParser()

# check argv
if len(sys.argv) < 2:
	print ("* missing argv")

# generate quiry
for item in sys.argv[1:]:
	quiry = quiry + item + "+"
#if debug: print (quiry)

response = requests.get (quiry)
#print (response.text)
parser.feed (response.text)
