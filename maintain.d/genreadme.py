#!/usr/bin/python3
import re
import sys
import os
import glob
import json
import string
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
configure = open('maintain.d/genreadme.conf', 'r').readlines()
configure = [ entry for entry in configure if len(re.findall('^#.*', entry))==0 ] # support comments in config file
configure = [ entry for entry in configure if len(entry.strip())>0 ] # remove empty lines
configure_sections, configure_omit, configure_normal = [], [], []
for line in configure:
    if len(re.findall('^%.*', line))!=0:
        configure_sections.append(line.replace('%','').strip())
    elif (len(line.strip().split('|'))==1 and len(re.findall('%',line))==0):
        configure_omit.append(line.strip())
    elif len(re.findall('|',line))>0:
        configure_normal.append(line.strip())
print('=> {} files found, {} configurations found'.format(len(filelist), len(configure)))

# filter-out files specified by single-column entries in config
print(' -> {} configure_omit'.format(len(configure_omit)))
for pattern in configure_omit:
  print('    > {}\t[omit]'.format(pattern))
  filelist = [ entry for entry in filelist if len(re.findall(pattern.strip(), entry))==0 ]
for patternplusxxx in configure_normal:
  filelist = [ entry for entry in filelist if len(re.findall(patternplusxxx.split('|')[0].strip(), entry))==0 ]

# generate sections
print(' -> {} configure_sections'.format(len(configure_sections)))
jsections = {} # dict{tag: list[ comment/url pairs ]}
for line in configure_normal:
    pattern, tag_comment = line.split('|')
    tag, comment = tag_comment.split(';')
    pattern, tag, comment = pattern.strip(), tag.strip(), comment.strip()
    print('   > ', pattern, '/', tag, '/', comment)
    if tag not in jsections.keys():
        jsections[tag] = []
    jsections[tag].append([pattern, string.capwords(comment)])
for k in jsections.keys():
    jsections[k] = sorted(jsections[k], key=(lambda x:x[1]))

# dump sections
print(jsections)
md = []
for tag in configure_sections:
    md.append("# {}\n".format(tag))
    md.append("\n")
    for pattern, comment in jsections[tag]:
        md.append('* [{}]({})  \n'.format(comment, pattern))
    md.append("\n")

print(''.join(md))

if len(filelist) > 10:
    pprint(filelist)
    print('-> {} files not configured.'.format(len(filelist)))
else:
    pprint(filelist)
    print('-> {} files not configured.')

# write file
headlines = [ line for line in open('maintain.d/README.md.head').readlines() ]
taillines = [ line for line in open('maintain.d/README.md.tail').readlines() ]
with open('README.md', 'w+') as f:
    f.writelines(headlines)
    f.writelines(md)
    f.writelines(taillines)

