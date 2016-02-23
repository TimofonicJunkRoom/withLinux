#!/usr/bin/python3
# reference: https://github.com/cdluminate/withlinux

import glob
import hashlib

jpeglist = glob.glob('pool/*.jpg')

count = 0
for jpg in jpeglist:
    with open(jpg, 'rb') as f:
        pic = f.read()
    if len(pic) > 0:
        if pic[0]!=255 or pic[1]!=216: # SOI
            count = count + 1
            m = hashlib.md5(pic)
            print ("%07d"%count, "NON-JPEG:", jpg, "(%s,%s) %s"%(pic[0], pic[1], m.hexdigest()))
        else:
            if pic[-2]!=255 or pic[-1]!= 217: # EOI
                count = count + 1
                print ("%07d"%count, "EOI-MISSING:", jpg, "(%s,%s)"%(pic[-2], pic[-1]))
