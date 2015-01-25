#!/usr/bin/python3.4
# -*- coding: utf=8 -*-

import html
import html.parser
from html.parser import HTMLParser

class MinerHTMLParser (HTMLParser):
	def handle_starttag (self, tag, attrs):
		ret = ""
		ret = ret + "_stag: " + tag
		for attr in attrs:
			ret = ret +  "	attr: " +  str(attr)
		return ret
	def handle_endtag (self, tag):
		print ("_etag: ", tag)
	def handle_data (self, data):
		print ("_data: ", data)

def parsehtml(_s):
	a = []
	parser = MinerHTMLParser()
	a.append(parser.feed (_s))
	ff = open ("/tmp/a", 'w+')
	ff.write(str(a))
	ff.close()

# temp: parser html
#		parsehtml (html.unescape(jsond["items_html"]))


