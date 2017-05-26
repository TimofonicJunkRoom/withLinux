#!/usr/bin/env python3
import json
import sys

if not sys.argv[1]:
    raise Exception('where is input json file?')

def lsjson(jobj, cdepth=0):
    '''
    Walk the json object (dict, list) recursively
    When the jobj is a list, we assume the inner structure
    of its elements is the same.
    '''
    #print('  '*cdepth, type(jobj), cdepth)
    if isinstance(jobj, list):
        if len(jobj)!=0: # non-empty list
            sample = jobj[0]
            print('  '*cdepth + '[', type(jobj))
            lsjson(sample, cdepth+1)
            print('  '*cdepth + '] ...')
        else:
            print('  '*cdepth + '[]')
    elif isinstance(jobj, dict):
        if len(jobj.keys())!=0: # non-empty dict
            print('  '*cdepth + '{', type(jobj))
            for key in jobj.keys():
                sample = jobj[key]
                print('  '*cdepth + ':' + key)
                lsjson(sample, cdepth+1)
            print('  '*cdepth + '}')
        else:
            print('  '*cdepth + '{}')
    else:
        print('  '*cdepth, type(jobj))

j_content = json.loads(open(sys.argv[1], 'r').read())
lsjson(j_content)
