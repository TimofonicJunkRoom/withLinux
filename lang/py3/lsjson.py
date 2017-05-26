#!/usr/bin/env python3
# Show the structure of a given json file.
# Copyright © 2017 Lumin <cdluminate@gmail.com>
# MIT License

import json
import sys

class lsJson(object):

    def __init__(self, arg_indent_block='    '):
        # configure
        self.s_indent_block = arg_indent_block
        self.__version__ = '1'

    def __call__(self, jobj, cdepth=0, show_example=False):
        self.lsjson(jobj, cdepth, show_example)

    def _c(self, s, color):
        ''' <helper>
        colorize the given string by wrapping it with ANSI color sequence
        in: s: given string
               color: string indicating the color
        out: str: colorized version of string s
        '''
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

    def _type(self, obj):
        ''' <helper>
        alternative to built-in function type()
        in: obj
        out: string indicating the type of the given object
        '''
        if isinstance(obj, list):
            return 'list'
        elif isinstance(obj, dict):
            return 'dict'
        elif isinstance(obj, int):
            return 'int'
        elif isinstance(obj, str):
            return 'str'
        elif isinstance(obj, float):
            return 'float'
        else:
            return str(type(obj))

    def lsjson(self, jobj, cdepth=0, show_example=False):
        ''' <main interface>
        Walk the json object (dict, list) recursively
        When the jobj is a list, we assume the inner structure
        of its elements is the same.
        '''
        #print(self.s_indent_block*cdepth, type(jobj), cdepth)
        if isinstance(jobj, list):
            if len(jobj)!=0: # non-empty list
                sample = jobj[0]
                print(self._c(self.s_indent_block*cdepth + '['+ str(self._type(jobj)), 'red'))
                lsjson(sample, cdepth=cdepth+1, show_example=show_example)
                print(self._c(self.s_indent_block*cdepth + '... ]', 'red'))
            else:
                print(self._c(self.s_indent_block*cdepth + '[]', 'red'))
        elif isinstance(jobj, dict):
            if len(jobj.keys())!=0: # non-empty dict
                print(self._c(self.s_indent_block*cdepth + '{'+ str(self._type(jobj)), 'yellow'))
                for key in jobj.keys():
                    '''
                    if the sample is int or str, don't break line
                    '''
                    sample = jobj[key]
                    if isinstance(sample, int) or isinstance(sample, str):
                        endline = ''
                    else:
                        endline = '\n'
                    print(self._c(self.s_indent_block*(cdepth+1) + ':{:20s}'.format(key), 'cyan'), end=endline)
                    lsjson(sample, cdepth=cdepth+2, show_example=show_example)
                print(self._c(self.s_indent_block*cdepth + '}', 'yellow'))
            else:
                print(self._c(self.s_indent_block*cdepth + '{}', 'yellow'))
        elif isinstance(jobj, int) or isinstance(jobj, float):
            print(self._c(self.s_indent_block*(9-cdepth)+str(self._type(jobj)), 'green'))
            if show_example:
                print(self._c(self.s_indent_block*(cdepth-1) + '-> '+str(repr(jobj)), 'violet'))
        elif isinstance(jobj, str):
            print(self._c(self.s_indent_block*(9-cdepth)+str(self._type(jobj)), 'white'))
            if show_example:
                print(self._c(self.s_indent_block*(cdepth-1) + '-> '+str(repr(jobj)), 'violet'))
        else:
            print(self._c(self.s_indent_block*cdepth+ str(self._type(jobj)), 'unknown'))

if __name__=='__main__':
    # configure
    b_show_example=False
    lsjson = lsJson()

    # argument check
    if len(sys.argv)==1:
        #raise Exception('where is input json file?')
        print(lsjson._c('where is input json file?', 'red'))
        exit(1)
    if len(sys.argv)==3 and sys.argv[2]=='-e': # show example
        b_show_example = True

    # main
    print(lsjson._c('=> lsjson '+sys.argv[1], 'violet'))
    try:
        j_content = json.loads(open(sys.argv[1], 'r').read())
        lsjson(j_content, show_example=b_show_example)
    except json.decoder.JSONDecodeError as e:
        print(lsjson._c('=> invalid or malformed file', 'red'))
        exit(2)
    except FileNotFoundError as e:
        print(lsjson._c('=> file not found', 'red'))
        exit(4) 
