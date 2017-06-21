#!/usr/bin/python3
import re
import sys
import os
import glob
from pprint import pprint

'''
genreadme.conf File Protocol

#                      Comment line
pattern                Python RE pattern of file (paths) to be omitted
'''

# glob files and read configuration
filelist  = glob.glob('**/*', recursive=True)
configure = open('readme.conf', 'r').readlines()
configure = [ entry for entry in configure if len(re.findall('^#.*', entry))==0 ] # support comments in config file
print('=> {} files found, {} configurations found'.format(len(filelist), len(configure)))

# filter-out files specified by single-column entries in config
configure_omit = [ line for line in configure if len(line.strip().split('|'))==1 ]
print(' -> {} configure_omit'.format(len(configure_omit)))
for pattern in configure_omit:
  print('    > {}\t[omit]'.format(pattern.strip()))
  filelist = [ entry for entry in filelist if len(re.findall(pattern.strip(), entry))==0 ]

pprint(filelist)
