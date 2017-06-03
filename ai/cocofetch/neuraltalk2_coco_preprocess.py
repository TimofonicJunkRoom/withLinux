#!/usr/bin/python3
# The original version of this file is from neuraltalk2.
import json
import os

with open('annotations/captions_train2014.json', 'r') as f:
    train_raw = f.read()
with open('annotations/captions_val2014.json', 'r') as f:
    val_raw = f.read()

train = json.loads(train_raw)
val   = json.loads(val_raw)

# combine all images and annotations together
imgs = val['images'] + train['images']
annots = val['annotations'] + train['annotations']

# for efficiency lets group annotations by image
itoa = {}
for a in annots:
    imgid = a['image_id']
    if not imgid in itoa: itoa[imgid] = []
    itoa[imgid].append(a)

# create the json blob
out = []
for i,img in enumerate(imgs):
    imgid = img['id']
    
    # coco specific here, they store train/val images separately
    loc = 'train2014' if 'train' in img['file_name'] else 'val2014'
    
    jimg = {}
    jimg['file_path'] = os.path.join(loc, img['file_name'])
    jimg['id'] = imgid
    
    sents = []
    annotsi = itoa[imgid]
    for a in annotsi:
        sents.append(a['caption'])
    jimg['captions'] = sents
    out.append(jimg)
    
json.dump(out, open('coco_raw.json', 'w'))
