#!/usr/bin/env python3
# Show the structure of a given json file.
# Copyright © Lumin <cdluminate@gmail.com>
# MIT License

import json
import sys

# configure
s_indent_block = '    '
b_show_example = False

# helper functions
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
        c = esc+'35;1m' # for example and special use
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

def lsjson(jobj, cdepth=0, show_example=False):
    '''
    Walk the json object (dict, list) recursively
    When the jobj is a list, we assume the inner structure
    of its elements is the same.
    '''
    #print(s_indent_block*cdepth, type(jobj), cdepth)
    if isinstance(jobj, list):
        if len(jobj)!=0: # non-empty list
            sample = jobj[0]
            print(_c(s_indent_block*cdepth + '['+ str(_type(jobj)), 'red'))
            lsjson(sample, cdepth=cdepth+1, show_example=show_example)
            print(_c(s_indent_block*cdepth + '... ]', 'red'))
        else:
            print(_c(s_indent_block*cdepth + '[]', 'red'))
    elif isinstance(jobj, dict):
        if len(jobj.keys())!=0: # non-empty dict
            print(_c(s_indent_block*cdepth + '{'+ str(_type(jobj)), 'yellow'))
            for key in jobj.keys():
                '''
                if the sample is int or str, don't break line
                '''
                sample = jobj[key]
                if isinstance(sample, int) or isinstance(sample, str):
                    endline = ''
                else:
                    endline = '\n'
                print(_c(s_indent_block*(cdepth+1) + ':{:20s}'.format(key), 'cyan'), end=endline)
                lsjson(sample, cdepth=cdepth+2, show_example=show_example)
            print(_c(s_indent_block*cdepth + '}', 'yellow'))
        else:
            print(_c(s_indent_block*cdepth + '{}', 'yellow'))
    elif isinstance(jobj, int):
        print(_c(s_indent_block*(9-cdepth)+str(_type(jobj)), 'green'))
        if show_example:
            print(_c(s_indent_block*(cdepth-1) + '-> '+str(repr(jobj)), 'violet'))
    elif isinstance(jobj, str):
        print(_c(s_indent_block*(9-cdepth)+str(_type(jobj)), 'white'))
        if show_example:
            print(_c(s_indent_block*(cdepth-1) + '-> '+str(repr(jobj)), 'violet'))
    else:
        print(_c(s_indent_block*cdepth+ str(_type(jobj)), 'unknown'))

# argument check
if not sys.argv[1]:
    raise Exception('where is input json file?')
if len(sys.argv)==3 and sys.argv[2]=='-e': # show example
    b_show_example = True

# main
print(_c('=> lsjson '+sys.argv[1], 'violet'))
j_content = json.loads(open(sys.argv[1], 'r').read())
lsjson(j_content, show_example=b_show_example)
