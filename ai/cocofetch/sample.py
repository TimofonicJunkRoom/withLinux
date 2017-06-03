#!/usr/bin/python3
import json
j = json.loads(open('annotations/captions_train2014.json').read())

annotations = j['annotations']
imagel = [ j['images'][i] for i in [0, 1, 2] ]
print ('sample id')
idl = [ imagel[i]['id'] for i in range(len(imagel)) ]
print (idl)

d = {}
for eachid in idl:
  sents = []
  for eachsent in annotations:
    if eachsent['image_id'] == eachid:
      sents.append(eachsent['caption'])
  d[eachid] = sents

for eachkey in d.keys():
  for eachsent in d[eachkey]:
    print ('%s: %s'%(eachkey, eachsent))
