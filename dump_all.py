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
with open('coco_all_sents.txt', 'w+') as f:
  for eachkey in d.keys():
    for eachsent in d[eachkey]:
      #print ('%s: %s'%(eachkey, eachsent))
      f.write('%s: %s\n'%(eachkey, eachsent.strip()))

print('validation: out-of-order raw sents')
with open('coco_all_sents.val.txt', 'w+') as f:
  for eachsent in annotationsall:
    f.write('%s: %s\n'%(eachsent['image_id'], eachsent['caption'].strip()))

os.system('''nl coco_all_sents.txt | tail -n1 ''')
os.system('''nl coco_all_sents.val.txt | tail -n1 ''')
