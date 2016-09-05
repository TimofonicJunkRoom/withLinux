#!/usr/bin/python3
import os
import glob

DEBUG=False

# Glob all files
flist_ = glob.glob('./**', recursive=True)
flist = []
for item in flist_:
  if os.path.isdir(item):
    pass
  elif os.path.islink(item):
    pass
  elif os.path.isfile(item):
    flist.append(item)
del flist_
print ('{} files globbed.'.format(len(flist)))

# Make statistics
lang = {}
for item in flist:
  # probe language
  if '.' in os.path.basename(item):
    needle = os.path.basename(item).split('.')[-1]
  elif 'Makefile' == os.path.basename(item):
    needle = 'Makefile'
  else:
    needle = 'plain'
  # increase counter
  if needle in lang:
    lang[needle] = lang[needle] + 1
  else:
    lang[needle] = 1
  if DEBUG: print ('processing {} / {} / {}'.format(item, needle, lang[needle]))

# Dump dictionary
maxkeylen = 0
for item in lang.keys():
  if len(item) > maxkeylen:
    maxkeylen = len(item)
lang = sorted(lang.items(), key = lambda d: d[1])
for (key, value) in lang:
  print ((' {:'+str(maxkeylen)+'s} : {}').format(key, value))
