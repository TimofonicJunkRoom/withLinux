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
config_count = 1


def do_search_1 (_keyword):
	url = "https://twitter.com/i/search/timeline?q=" + _keyword
	if debug: print ("* [do_search_1] {}".format(url).replace('\n', ''))
	page = requests.get (url)
	return page.text


def do_search_2 (_keyword, _cursor):
	"""
	send the request to server
	return the page.text server returned here
	"""
	url = "https://twitter.com/i/search/timeline?q=" + _keyword + "&scroll_cursor=" + _cursor
	if debug: print ("* [do_search_2] {}".format(url).replace('\n', ''))
#	page = requests.get (url, proxies = proxies1)
	page = requests.get (url)
	return page.text


def mine (_keyword, _count):
	count = 1
	cursor = ""
# check ./pool/{_keyword} directory
	if not os.path.exists ("pool"):
		os.makedirs ("pool")
	if not os.path.exists ("pool/"+_keyword):
		os.makedirs ("pool/"+_keyword)
# search
	while (count <= _count):
		print ("* [main] Search '{}' for the {} (th) time.".format(_keyword, count))
		# get the raw page and save to archive
		if count == 1:
			page = do_search_1 (_keyword)
		else:
			page = do_search_2 (_keyword, cursor)
		with open ("pool/{}/{}".format(_keyword, str(count)), "w+") as _f:
			_f.write (page)
		# save html included in json
		jsond = json.loads(page)
		if not "items_html" in jsond:
			print ("unexpected error: can't find item_html, abort.")	
		with open ("pool/{}/{}.html".format(_keyword, str(count)), "w+") as _f:
			_f.write (jsond["items_html"])
		# parse cursor
		if "scroll_cursor" in jsond:
			cursor = jsond["scroll_cursor"]
		else:
			print ("unexpected error: can't find scroll_cursor, abort.")
			exit (1)
		if debug>2: print ("* [main] Generated NEW Cursor :\n\t{}".format(cursor))
		count = count + 1
	
def main():
# check config file
		if not os.path.exists ("mine.list"):
			print ("missing mine.list")
			exit (1)
# open config and start mine!
		config_file = open ("mine.list")
		for words in config_file.readlines():
			if words[0] == '#': continue
			mine (words.replace('\n', '').replace(' ', '+'), config_count)
		config_file.close()	


if __name__ == "__main__":
	main()
