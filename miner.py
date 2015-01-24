#!/usr/bin/python3.4

# cdluminate@163.com

#proxies1 = {
#	"http": "http://127.0.0.1:8087",
#	"https": "https://127.0.0.1:8087"
#}

import os
import requests
import json

debug = 1
config_count = 2

def do_search_1 (_keyword):
	url = "https://twitter.com/i/search/timeline?q=" + _keyword
	if debug: print ("* [do_search] KEY : {}, url : {}".format(_keyword, url).replace('\n', ''))
	page = requests.get (url)
	return page.text

def do_search_2 (_keyword, _cursor):
	"""
	send the request to server
	return the page.text server returned here
	"""
	url = "https://twitter.com/i/search/timeline?q=" + _keyword + "&scroll_cursor=" + _cursor
	if debug: print ("* [do_search] KEY : {}, url : {}".format(_keyword, url).replace('\n', ''))
#	page = requests.get (url, proxies = proxies1)
	page = requests.get (url)
	return page.text

def mine (_keyword, _count):
	count = _count
	cursor = ""
# check ./pool/{_keyword} directory
	if not os.path.exists ("pool"):
		os.makedirs ("pool")
	if not os.path.exists ("pool/"+_keyword):
		os.makedirs ("pool/"+_keyword)
# search
	while (count >= 0):
		print ("* [main] Searching, {} time(s) left.".format(count))
		# get page and save to archive
		if count == _count:
			page = do_search_1 (_keyword)
		else:
			page = do_search_2 (_keyword, cursor)
		_f = open ("pool/{}/{}".format(_keyword, str(count)), "w+")
		_f.write (page)
		_f.close ()
		# parse cursor
		jsond = json.loads(page)
		if "scroll_cursor" in jsond:
			cursor = jsond["scroll_cursor"]
		else:
			print ("unexpected error: can't find scroll_cursor.")
			exit (1)
		if debug>2: print ("* [main] Generated NEW Cursor :\n\t{}".format(cursor))
		count = count - 1

def main():
# check config file
		if not os.path.exists ("mine.list"):
			print ("missing mine.list")
			exit (1)
# open config and start mine!
		config_file = open ("mine.list")
		for words in config_file.readlines():
				mine (words.replace('\n', '').replace(' ', '+'), config_count)
		config_file.close()	

main()
