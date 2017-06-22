#!/usr/bin/python3
import re
import sys
import os
import glob
import json
from pprint import pprint

'''
genreadme.conf File Protocol

#                      Comment line
pattern                Python RE pattern of file (paths) to be omitted
pattern | tag;comment  ...
% tag                  Generate section for the specified tag
'''

# glob files and read configuration
filelist  = glob.glob('**/*', recursive=True)
configure = open('genreadme.conf', 'r').readlines()
configure = [ entry for entry in configure if len(re.findall('^#.*', entry))==0 ] # support comments in config file
configure = [ entry for entry in configure if len(entry.strip())>0 ] # remove empty lines
configure_sections  = [ entry.replace('%','').strip() for entry in configure if len(re.findall('^%.*', entry))!=0 ]
configure_omit = [ line.strip() for line in configure if (len(line.strip().split('|'))==1 and len(re.findall('%',line))==0) ]
configure_normal = [ line.strip() for line in configure if len(re.findall('|',line))>0 ]
print('=> {} files found, {} configurations found'.format(len(filelist), len(configure)))

# filter-out files specified by single-column entries in config
print(' -> {} configure_omit'.format(len(configure_omit)))
for pattern in configure_omit:
  print('    > {}\t[omit]'.format(pattern))
  filelist = [ entry for entry in filelist if len(re.findall(pattern.strip(), entry))==0 ]

# generate sections
print(' -> {} configure_sections'.format(len(configure_sections)))
jsections = {} # dict{tag: list[ comment/url pairs ]}

#pprint(filelist)
