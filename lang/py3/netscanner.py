#!/usr/bin/python3
'''
"Ridiculously simple network scanner with curl"
It scans for SSH host in specified network range.
'''
from multiprocessing.dummy import Pool
from subprocess import Popen, PIPE
from sys import stdout

def scan(target):
  cmd = [ 'curl', '-m', '0.05', '-s', target ]
  print('scan:', ' '.join(cmd))
  proc = Popen(cmd, shell=False, stdout=PIPE)
  out, err = proc.communicate()
  if len(out) > 0:
    print('reply detected:', ' '.join(cmd))
    print(str(out.decode()))
  stdout.flush()

def gentargets(port):
  targets = []
  base = '10.170.{}.{}:{}'
  for i in range(1,256):
    for j in range(1, 256):
      targets.append( base.format(i,j, port) )
  return targets

if __name__ == '__main__':
  targets = gentargets(10022)
  with Pool(36) as tp:
    tp.map(scan, targets)
