#!/bin/sh
for item in $(ls *.out); do
  printf "%s\t" $item
  ./stub/filter.py $item | grep ROOT | nl | tail -n1
done
