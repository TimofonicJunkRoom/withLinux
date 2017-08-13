#!/usr/bin/python3

import urllib3 as UL3
import simplejson as json
from subprocess import call

# configure
howmany = 10
counter = 0

# do the first time search
http = UL3.PoolManager()
url = ('https://ajax.googleapis.com/ajax/services/search/images?' +
	   'v=1.0&q={}'.format('https://www.google.co.jp/intl/en_ALL/images/srpr/logo11w.png'))
response = http.request ('GET', url)
results = json.loads(response.data)
## now have some fun with the results...
for item in results['responseData']['results']:
	if 'unescapedUrl' in item:
		print ("{}".format(item['unescapedUrl']))
#call ("wget -P pool/ {}".format(item['unescapedUrl']), shell=True)

counter = 4

while counter < howmany:
	url = ('https://ajax.googleapis.com/ajax/services/search/images?' +
		   'v=1.0&q={0:s}&start={1:d}'.format('https://www.google.co.jp/intl/en_ALL/images/srpr/logo11w.png', counter))
	#		   'v=1.0&q=barack%20obama&start={0:d}'.format(counter))
	response = http.request ('GET', url)
	#print (response.data)
	results = json.loads(response.data)
	## now have some fun with the results...
	for item in results['responseData']['results']:
		if 'unescapedUrl' in item:
			print ("{}".format(item['unescapedUrl']))
#call ("wget -P pool/ {}".format(item['unescapedUrl']), shell=True)
	counter = counter + 4
