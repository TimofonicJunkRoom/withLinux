#!/usr/bin/python3
''' Split and parse all output from dump_all.py '''

import os
import sys

CP='../stanford-corenlp-full-2015-12-09/*'
command='java -cp "' + CP + '" -Xmx3g edu.stanford.nlp.pipeline.StanfordCoreNLP \
 -annotators tokenize,ssplit,pos,lemma,ner,parse,dcoref \
 -file %s \
 -outputFormat text 1>/dev/null 2>/dev/null'

destdir='coco_all_sents.st5.split'
partnum=5000

with open('coco_all_sents.st5.txt', 'r') as f:
  lines = f.readlines()

if not os.path.exists(destdir):
  os.mkdir(destdir)

cursor=0
count=0
part=0
buf=[]
while (cursor < len(lines)):
  buf.append(lines[cursor])
  count = count + 1
  cursor = cursor + 1
  if count >= partnum:
    with open(destdir + '/part' + str(part), 'w+') as f:
      f.write(''.join(buf))
      print(part+1, '/', len(lines)/partnum)
    count = 0
    part = part + 1
    buf.clear()

with open(destdir + '/part' + str(part), 'w+') as f:
  f.write(''.join(buf))
  print(part+1, '/', len(lines)/partnum)

os.sync()
sys.stdout.flush()

import math
from multiprocessing import Pool
def parse(part):
  os.system(command%(destdir+'/part'+str(part)))
  print('done part', part)
  sys.stdout.flush()
parts = list(range(math.ceil(len(lines)/partnum)))
print(len(parts), 'parts', parts[0], parts[-1])
print('mapping', len(parts), 'tasks to process pool')

pool = Pool(6)
pool.map(parse, parts)
pool.close()
pool.join()

print('done')
