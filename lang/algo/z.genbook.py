#!/usr/bin/python3
import os
import sys
import glob
from pprint import pprint
import shlex
from subprocess import call

rstbook = []
rstbook.append("""
========================
Auto-Generated Code Book
========================


:Author: Lumin 
:Date:   Oct 2017

.. contents::
   :depth: 3
..

""")

if __name__=='__main__':

    files = glob.glob('**')
    # remove myself
    files.pop(files.index(__file__))
    print('=> Found {} files'.format(len(files)))
    #pprint(files)

    # categories
    cppfiles = [ f for f in files if f.endswith('.cc') ]
    cppfiles.sort()
    print(' -> {} cpp files'.format(len(cppfiles)))
    pyfiles  = [ f for f in files if f.endswith('.py') ]
    pyfiles.sort()
    print(' -> {} py files'.format(len(pyfiles)))

    # append cpp part
    rstbook.append("""
C++ Part
========
""")
    for i, f in enumerate(cppfiles, 1):
        #print (' * adding {} ...'.format(f))
        rstbook.append("""
{}. ``{}``
{}

.. code:: cpp

""".format(i, f, '-'*( len(str(i)) + len(f) + 6 )))
        fo = open(f, 'r')
        for line in fo.readlines():
            rstbook.append('  ' + line)
        fo.close()

    # append python part
    rstbook.append("""
Python Part
===========
""")
    for i, f in enumerate(pyfiles, 1):
        rstbook.append("""
{}. ``{}``
{}

.. code:: python

""".format(i, f, '-'*( len(str(i)) + len(f) + 6 )))
        fo = open(f, 'r')
        for line in fo.readlines():
            rstbook.append('  ' + line)
        fo.close()

    # save, convert, cleanup
    with open(__file__ + '.rst', 'w+') as f:
        f.writelines(rstbook)
    print('=> Saved. Start to generate pdf ...')
    cmd_pandoc = 'pandoc -f rst -t latex z.genbook.py.rst -o z.genbook.py.pdf'
    cmd_cleanup = 'rm z.genbook.py.rst'
    print(' -> pandoc ...')
    call(shlex.split(cmd_pandoc))
    print(' -> clean up ...')
    call(shlex.split(cmd_cleanup))
    print('=> Done.')
