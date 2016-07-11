#!/usr/bin/python3
import json
import os
j = json.loads(open('annotations/captions_train2014.json').read())
jv = json.loads(open('annotations/captions_val2014.json').read())

annotations = j['annotations']
annotationsv = jv['annotations']
annotationsall = annotations + annotationsv
print('annotationsall', type(annotationsall), 'length', len(annotationsall))

imagel = [j['images'][i] for i in range(len(j['images']))]
idl = [ imagel[i]['id'] for i in range(len(imagel)) ]
assert(len(idl) == len(imagel))
print('train image', type(imagel), len(imagel))

imagelv = [jv['images'][i] for i in range(len(jv['images']))]
idlv = [ imagelv[i]['id'] for i in range(len(imagelv)) ]
assert(len(idlv) == len(imagelv))
print('val image', type(imagelv), len(imagelv))

idlall = sorted(idl + idlv)
print('idlall', type(idlall), len(idlall))

print('stage1: convert to dict')
d = {}
pcounter=0
rcounter=1
for eachsent in annotationsall:
  if pcounter > 30000:
    print('stage1: progress %.1f%%'%((100*rcounter)/len(annotationsall)))
    pcounter = 0
  cur_image_id = eachsent['image_id']
  if not cur_image_id in d:
    d[cur_image_id] = []
  d[cur_image_id].append(eachsent['caption'])
  pcounter = pcounter + 1
  rcounter = rcounter + 1
  
print('stage2: writing text file')
with open('coco_all_sents.st2.txt', 'w+') as f:
  for eachkey in d.keys():
    for eachsent in d[eachkey]:
      if len(eachsent.strip()) == 0:
        continue
      eachsent = eachsent.strip().replace('\n', ' ').strip()
      if eachsent[-1]=='.' or eachsent[-1]=='?' or eachsent[-1]=='!':
        tmp = list(eachsent)
        tmp[-1] = ' '
        eachsent = ''.join(tmp)
      eachsent = eachsent.strip().replace('.', ',').strip()
      eachsent = eachsent.strip().replace('?', ',').strip()
      eachsent = eachsent.strip().replace('!', '').strip()
      eachsent = eachsent.strip() + ' .'
      f.write('%s: %s\n'%(eachkey, eachsent))

os.system('''nl coco_all_sents.st2.txt | tail -n1 ''')
with open('coco_all_sents.st2.txt', 'r') as f:
  buf = f.read()
  d = {}
  #print(type(buf))
  count = 0
  for char in buf:
    if char == '.' or char == '?':
      count = count + 1
    if not char in d:
      d[char] = 0
    d[char] = d[char] + 1
  print('note:', count, 'dots found')
  print(d.keys())
  assert(count == 616767)

print('stage3: strip coco_all_sents.txt')
with open('coco_all_sents.st2.txt', 'r') as fi:
  with open('coco_all_sents.st3.txt', 'w+') as fo:
    lines = fi.readlines()
    for (k,line) in enumerate(lines):
      # tester
      line = line.strip()
      try:
        needle = line[-1]
      except IndexError:
        print('TROUBLEMAKER:', line, len(line))
        del line
        continue
      # write
      if len(line.strip()) == 0:
        continue
      else:
        n = fo.write('%s\n'%line)
        assert(n>0)

os.system('''nl coco_all_sents.st3.txt | tail -n1 ''')

print('stage4: performing check against coco_all_sents.st3.txt')
with open('coco_all_sents.st3.txt', 'r') as f:
  lines = f.readlines()
  assert(len(lines) == 616767)
  for (k,line) in enumerate(lines):
    line = line.strip()
    if len(line) == 0:
      print('warning:', k, '-th line empty')
  for (k,line) in enumerate(lines):
    if line.strip()[-1] != '.':
      print('warning:', k, '-th line missing period:', line.strip()[-1])
  for (k, line) in enumerate(lines):
    try:
      needle = line[-1]
    except IndexError:
      print('trouble line not removed!')

os.system('''nl coco_all_sents.st3.txt | tail -n1 ''')

print('stage5: strip image_num')
with open('coco_all_sents.st3.txt', 'r') as fi:
  with open('coco_all_sents.st5.txt', 'w+') as fo:
    lines = fi.readlines()
    for line in lines:
      line = line.strip().split(' ')[1:] # remove heading image_num
      fo.write('%s\n'%(' '.join(line)))

os.system('''nl coco_all_sents.st5.txt | tail -n1 ''')
