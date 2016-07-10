#!/usr/bin/python3
import json
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
counter=0
for eachid in idlall:
  if counter > 32:
    print('stage1: progress %.2f%%'%((100*eachid)/idlall[-1]))
    counter = 0
  sents = []
  for eachsent in annotations:
    if eachsent['image_id'] == eachid:
      sents.append(eachsent['caption'])
    else:
      pass
  d[eachid] = sents
  counter = counter + 1

print('stage2: writing text file')
with open('coco_all_sents.txt', 'w+') as f:
  for eachkey in d.keys():
    for eachsent in d[eachkey]:
      #print ('%s: %s'%(eachkey, eachsent))
      f.write('%s: %s\n'%(eachkey, eachsent))
