#!/usr/bin/env python3
# cxdb.py, create xapian db
#
# reference: python3-xapian/html/examples.html#simpleindex-py

import os
import string
import sys
import traceback

import xapian # apt install python3-xapian

class Spinner(object):
    def __init__(self):
        self._counter = 0
        self._spinner = list('-\|/')
    def spin(self):
        cursor = self._spinner[self._counter % len(self._spinner)]
        self._counter += 1
        print('\b'+cursor, end='')
        sys.stdout.flush()
spinner = Spinner()

# http://stackoverflow.com/questions/898669/how-can-i-detect-if-a-file-is-binary-non-text-in-python
textchars = bytearray({7,8,9,10,12,13,27} | set(range(0x20, 0x100)) - {0x7f})
is_binary_string = lambda bytes: bool(bytes.translate(None, textchars))

try:
    xdb = xapian.WritableDatabase("__xdb__", xapian.DB_CREATE_OR_OVERWRITE)

    indexer = xapian.TermGenerator()
    stemmer = xapian.Stem("english")
    indexer.set_stemmer(stemmer)

    # scan the project
    counter_indexed = 0
    pathlist_indexed = []
    for dirpath, _, filenames in os.walk("."):
        if ".git" in dirpath or "__xdb__" in dirpath:
            continue
        for filename in filenames:
            cursor = os.path.join(dirpath, filename)

            # skip non-plain files
            with open(cursor, 'rb') as cursor_file:
                cursor_content = cursor_file.read()
                if is_binary_string(cursor_content):
                    print("D: skip non-plain file {}".format(cursor))
                    continue
            #print("I: {:04d} : processing {}".format(counter_indexed, cursor))
            spinner.spin()

            try:
                doc = xapian.Document()
                with open(cursor, 'r') as cursor_file:
                    cursor_document = cursor_file.read()
                   
                doc.set_data(cursor_document)

                indexer.set_document(doc)
                indexer.index_text(cursor_content)

                xdb.add_document(doc)

                counter_indexed = counter_indexed + 1
                pathlist_indexed.append(cursor)
            except:
                print("W: skip problematic file {}".format(cursor))
                pass
    with open('__xdb__/path.list', 'w+') as f:
        f.write('\n'.join(pathlist_indexed))
    print("I: path list saved.")

except Exception as e:
    print("Exception: {}".format(e))
    traceback.print_exc()
    sys.exit(1)
