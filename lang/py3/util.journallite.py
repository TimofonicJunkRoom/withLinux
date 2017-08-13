#!/usr/bin/python3
import os
import sys
import subprocess

if len(sys.argv)!=2:
  raise Exception('missing operand')

# round 0: get recall
print('I: round0: get recall')
proc = subprocess.Popen(['grep', '-i', '-A3', 'recall', '{}'.format(sys.argv[1].strip())],
                        stdout=subprocess.PIPE)
out, err = proc.communicate()
with open('{}.recall'.format(sys.argv[1].strip()), 'w+') as f:
  f.write(str(out.decode()))
print('I: wrote recall')

# round 1: get bottom 50K lines
print('I: get bottom line number')
with open(sys.argv[1].strip(), 'r') as f:
  line = f.readline(); k = 1 # small buffer requirement
  linequeue = [ line ]
  while len(line)>0:
    #print(k, line.strip())
    if k%1000000==0: print('I: round1: read {} lines'.format(k))
    line = f.readline(); k = k + 1
    linequeue.append(line)
  if len(linequeue) > 500000:
    del linequeue[:-500000]
  bottom_line_num = k
  print('I: found {} lines in total'.format(bottom_line_num))

  # write lite version
  print('I: write lite version with 50K lines')
  with open(sys.argv[1].strip()+'.lite', 'w+') as target:
    target.writelines(linequeue)

print('I: done')
