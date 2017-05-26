#!/usr/bin/env python3
import json
import sys

if not sys.argv[1]:
    raise Exception('where is input json file?')

def _c(s, color):
    esc = '\x1b['
    restore = esc + ';m'
    if color=='red':
        c = esc+'31;1m' # red for list
    elif color=='green':
        c = esc+'32;1m' # green for int
    elif color=='yellow':
        c = esc+'33;1m' # yellow for dict
    elif color=='blue':
        c = esc+'34;1m' # blue for unknown
    elif color=='cyan':
        c = esc+'36;1m' # cyan for dict key
    elif color=='white':
        c = esc+'37;1m' # white for string
    elif color=='violet':
        c = esc+'35;1m'
    else:
        c = ''
    return c+s+restore

def _type(obj):
    if isinstance(obj, list):
        return 'list'
    elif isinstance(obj, dict):
        return 'dict'
    elif isinstance(obj, int):
        return 'int'
    elif isinstance(obj, str):
        return 'str'
    else:
        return str(type(obj))

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
            print(_c('  '*cdepth + '['+ str(_type(jobj)), 'red'))
            lsjson(sample, cdepth+1)
            print(_c('  '*cdepth + '] ...', 'red'))
        else:
            print(_c('  '*cdepth + '[]', 'red'))
    elif isinstance(jobj, dict):
        if len(jobj.keys())!=0: # non-empty dict
            print(_c('  '*cdepth + '{'+ str(_type(jobj)), 'yellow'))
            for key in jobj.keys():
                '''
                if the sample is int or str, don't break line
                '''
                sample = jobj[key]
                if isinstance(sample, int) or isinstance(sample, str):
                    endline = ''
                else:
                    endline = '\n'
                print(_c('  '*(cdepth+1) + ':{:20s}'.format(key), 'cyan'), end=endline)
                lsjson(sample, cdepth+2)
            print(_c('  '*cdepth + '}', 'yellow'))
        else:
            print(_c('  '*cdepth + '{}', 'yellow'))
    elif isinstance(jobj, int):
        print(_c('  '*(9-cdepth)+str(_type(jobj)), 'green'))
    elif isinstance(jobj, str):
        print(_c('  '*(9-cdepth)+str(_type(jobj)), 'white'))
    else:
        print(_c('  '*cdepth+ str(_type(jobj)), 'unknown'))

print(_c('=> lsjson '+sys.argv[1], 'violet'))
j_content = json.loads(open(sys.argv[1], 'r').read())
lsjson(j_content)
