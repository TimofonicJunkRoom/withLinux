#!/usr/bin/python3
# Copyright (C) 2016 Lumin Zhou
# MIT License

import json
import subprocess
import sys
import os

debug = 1

def main ():
    if len(sys.argv) != 2:
        print ("missing argv[1]")

    if debug: print (sys.argv)

    with open(sys.argv[1], 'r') as f:
        content = f.read()
        d = json.loads(content)
        if debug: print (d.keys())
        '''
        ['images', 'licenses', 'annotations', 'info']
        '''
        if debug: print (d["images"][0])
        middle = int(len(d['images'])/2)
        print ("total", len(d['images']), "middle", middle)

        left = {}
        left['licenses'] = d['licenses']
        left['annotations'] = d['annotations']
        left['info'] = d['info']
        left['images'] = d['images'][0:middle]
        with open(sys.argv[1] + ".left", "w+") as fl:
            fl.write(json.dumps(left))
            print ("split: part 1", sys.argv[1]+".left")

        right = {}
        right['licenses'] = d['licenses']
        right['annotations'] = d['annotations']
        right['info'] = d['info']
        right['images'] = d['images'][middle:]
        with open(sys.argv[1] + ".right", "w+") as fr:
            fr.write(json.dumps(right))
            print ("split: part 2", sys.argv[1]+".right")

main ()
