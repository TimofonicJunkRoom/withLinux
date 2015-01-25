#!/usr/bin/python3.4
# -*- coding: utf=8 -*-

import html
import html.parser
from html.parser import HTMLParser

depth = 0

class MinerHTMLParser (HTMLParser):
	depth = 0
	def handle_starttag (self, tag, attrs):
		depth = depth + 1
		print ("\t"*depth + "_stag: ", tag)
		for attr in attrs:
			print ("	attr: ", str(attr))
	def handle_endtag (self, tag):
		depth = depth - 1
		print ("\t"*depth + "_etag: ", tag)
	def handle_data (self, data):
		depth = depth
		print ("\t"*(depth+1) + str(data))


def main():
	f = open ("pool/linux/1.html", 'r')
	page = f.read()
	f.close()
#print (page)
	parser = MinerHTMLParser()
#	parser = HTMLParser()
	parser.feed (page)

main()
