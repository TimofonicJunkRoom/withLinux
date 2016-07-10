#!/usr/bin/python3
import json
j = json.loads(open('annotations/captions_train2014.json').read())

annotations = j['annotations']

imagel = [j['images'][i] for i in range(len(j['images']))]

idl = [ imagel[i]['id'] for i in range(len(imagel)) ]
idl = sorted(idl)
print ('sorted list len ', len(idl))

d = {}
for eachid in idl:
  print('progress %f%%'%((100*eachid)/idl[-1]))
  sents = []
  for eachsent in annotations:
    if eachsent['image_id'] == eachid:
      sents.append(eachsent['caption'])
  d[eachid] = sents

with open('coco_all_sents.txt', 'w+') as f:
  for eachkey in d.keys():
    for eachsent in d[eachkey]:
      #print ('%s: %s'%(eachkey, eachsent))
      f.write('%s: %s\n'%(eachkey, eachsent))
