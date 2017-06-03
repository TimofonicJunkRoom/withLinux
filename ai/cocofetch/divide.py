#!/usr/bin/python3
''' Split and parse all output from dump_all.py '''

import os

CP='../../treelstm/data/stanford-corenlp-full-2015-12-09/*'
command='java -cp "' + CP + '" -Xmx4g edu.stanford.nlp.pipeline.StanfordCoreNLP \
 -annotators tokenize,ssplit,pos,lemma,ner,parse,dcoref \
 -file %s \
 -outputFormat text'

destdir='coco_all_sents.st5.split'

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
  if count >= 1000:
    with open(destdir + '/part' + str(part), 'w+') as f:
      f.write(''.join(buf))
      os.sync()
      print(part+1, '/', len(lines)/1000)
    os.system(command%(destdir+'/part'+str(part)))
    count = 0
    part = part + 1
    buf.clear()

with open(destdir + '/part' + str(part), 'w+') as f:
  f.write(''.join(buf))
  os.sync()
  print(part+1, '/', len(lines)/1000)
os.system(command%(destdir+'/part'+str(part)))

print('done')
