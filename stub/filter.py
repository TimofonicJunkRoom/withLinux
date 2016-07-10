#!/usr/bin/python3
'''
filters corenlp text output for trees
'''
import sys

with open(sys.argv[1], 'r') as f:
  lines = f.readlines()
  flag = 0
  for line in lines:
    if line[0:5] == '(ROOT':
      flag = 1
      print(line, end='')
    elif line[0] == ' ' and flag == 1:
      print(line, end='')
    elif line[0] != ' ':
      flag = 0
