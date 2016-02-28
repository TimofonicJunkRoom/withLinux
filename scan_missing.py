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
        count = 0
        for item in d["images"]:
            if not os.path.exists("pool/" + item["file_name"]):
                count = count + 1
                print ("%05d"%count, "missing", ""+item["file_name"])

main ()
