#!/usr/bin/python3

import json
import subprocess
import sys
import os

debug = 1

coco_url_key = "coco_url"
flickr_url_key = "flickr_url"
using_url = flickr_url_key
#using_url = coco_url_key

def main ():
    if len(sys.argv) != 2:
        print ("missing argv[1]")

    if debug: print (sys.argv)

    with open(sys.argv[1], 'r') as f:
        content = f.read()
        d = json.loads(content)
        if debug: print (d.keys())
        if debug: print (d["images"][0])
        for item in d["images"]:
            if os.path.exists("pool/" + item["file_name"]):
                print ("skip", item[using_url], ":", item["file_name"])
            else:
                print ("download", item[using_url], "as", item["file_name"])
                subprocess.call (["wget", item[using_url], "-O", "pool/" + item["file_name"]])

main ()
