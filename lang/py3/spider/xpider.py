#!/usr/bin/python3
'''
automatic image downloader from image.Baidu.com

1. demand analysis
2. regex design
3. write python

@reference http://urllib3.readthedocs.io/en/latest/
@reference http://selenium-python.readthedocs.io/
@reference <<Web Scraping with Python>>
'''
import urllib.parse
import urllib3
import re
import os
import magic
import time
from selenium import webdriver

# configure
url_template="http://image.baidu.com/search/index" + \
  "?tn=baiduimage&ipn=r&cl=2&lm=-1&st=-1&fm=result&fr=&sf=1" + \
  "&pv=&ic=0&nc=1&z=&se=1&showtab=0&fb=0&width=&height=&face=0" + \
  "&istype=2&ie=utf-8&word="
keyword=u'暴走表情'
goal_image_num = 50

# initialization
re_pic = re.compile('"objURL":"(.*?)",')
pm = urllib3.PoolManager()
ms = magic.open(magic.NONE)
ms.load()
driver = webdriver.Chrome('/usr/lib/chromium/chromedriver')

# helpers
def download(url):
  if not os.path.exists('pool'):
    print('create pool directory')
    os.mkdir('pool')
  pic = pm.request('GET', url)
  basename = os.path.basename(urllib.parse.unquote(url))
  path = 'pool/' + basename
  filemagic = ms.buffer(pic.data).split(',')[0]
  #print(pic.data)
  with open(path, 'wb') as f:
    print('write', filemagic, '->', path, '[', pic.status, ']')
    f.write(pic.data)
  return pic.status

def main():

  # --[ fetch the initial html page
  url = url_template + urllib.parse.quote(keyword)
  print('fetch and parse', ':', urllib.parse.unquote(url))
  driver.get(url)
  time.sleep(3)
  if not u'百度图片' in driver.title:
    print('Error: could not access', url)
    exit(1)
  #print(driver.page_source)

  # --[ load enough images
  imgitems = driver.find_elements_by_class_name('imgitem')
  while len(imgitems) < goal_image_num:
    # http://selenium-python.readthedocs.io/faq.html#how-to-scroll-down-to-the-bottom-of-a-page
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    imgitems = driver.find_elements_by_class_name('imgitem')
    print('waiting for more result ... (%d/%d)'%
      (len(imgitems), goal_image_num))
    time.sleep(2)
  print('got %d imgitems, %d required'%(len(imgitems), goal_image_num))

  # --[ extract image urls
  # imgitem.get_attribute('xxx')
  # 'class' 'style' 'data-object' 'data-objurl'
  imgitems_urls = []
  for each in imgitems:
    data_objurl = each.get_attribute('data-objurl')
    #data_thumburl = each.get_attribute('data-thumburl')
    imgitems_urls.append(data_objurl)
  #for each in imgitems_urls:
  #  print(each)

  # --[ close driver
  driver.close()

  # --[ invoke downloader against all fetched urls
  print(len(imgitems_urls), 'image urls parsed, starting to download ...')

  res = list(map(download, imgitems_urls))
  print(res)

  # --[ oh yes
  print('all pictures you requested are in place')

main()
