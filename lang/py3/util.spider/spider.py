#!/usr/bin/python3
'''
automatic image downloader

1. demand analysis
2. regex design
3. write python

@reference http://urllib3.readthedocs.io/en/latest/

TODO:

* upgrade and use webdriver
'''
import urllib.parse
import urllib3
import re
import os
import magic

# configure
url_template="http://image.baidu.com/search/index" + \
  "?tn=baiduimage&ipn=r&cl=2&lm=-1&st=-1&fm=result&fr=&sf=1" + \
  "&pv=&ic=0&nc=1&z=&se=1&showtab=0&fb=0&width=&height=&face=0" + \
  "&istype=2&ie=utf-8&word="
keyword=u'暴走表情'

# initialization
re_pic = re.compile('"objURL":"(.*?)",')
pm = urllib3.PoolManager()
ms = magic.open(magic.NONE)
ms.load()

# helpers
def download(url):
  if not os.path.exists('pool'):
    print('create pool directory')
    os.mkdir('pool')
  pic = pm.request('GET', url)
  basename = os.path.basename(urllib.parse.unquote(url))
  path = 'pool/' + basename
  filemagic = ms.buffer(pic.data).split(',')[0]
  print('write', filemagic, '->', path)
  #print(pic.data)
  with open(path, 'wb') as f:
    f.write(pic.data)
  return pic.status

def main():
  url = url_template + urllib.parse.quote(keyword)
  print('fetch and parse', ':', urllib.parse.unquote(url))
  res = pm.request('GET', url)
  #print(res.status, ':', url)
  print('http status', res.status)
  if res.status != 200:
    print('html page fetch failure')
  pic_urls = re_pic.findall(str(res.data))
  #print(pic_urls)
  print(len(pic_urls), 'images parsed')
  print('starting to download')
  #res = list(map(download, pic_urls)) # we don't download all, this is only a demo
  res = list(map(download, pic_urls[0:5]))
  print(res)

main()
