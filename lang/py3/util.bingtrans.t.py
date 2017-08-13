#!/usr/bin/python3.4
# translate argv with bing
# 2015 Lumin Zhou

'''
# bingtrans
[py3] bing translation shortcut in terminal  
  
It translates the argv you'd given, that's all.

### example  
```
$ python3.4 t.py python
```

### hint
add this into bashrc:
```
t () {
  python3.4 some/dir/t.py $@
}
```
then you can just call the script with
```
$ t <your keyword or sentense>
```
'''

import sys
import requests
from html.parser import HTMLParser

class DictParser (HTMLParser):
	def handle_starttag (self, tag, attrs):
		for attr, data in attrs:
			if attr == "content":
				if data.find("必应词典") == 0:
					print (data.replace("，", "，\n")
					       .replace(".", ".\n")
						   .replace("；", "；\n")
						   .replace("：", "：\n")
						  )
# configure
baseurl = "https://www.bing.com/dict/search?q="
quiry   = baseurl
parser  = DictParser()

# check argv
if len(sys.argv) < 2:
	print ("* missing argv")

# generate quiry
for item in sys.argv[1:]:
	quiry = quiry + item + "+"

response = requests.get (quiry)
parser.feed (response.text)

