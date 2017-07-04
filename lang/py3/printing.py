#!/usr/bin/python3

import os
import sys

print('{:5d} {:7d}'.format(1, 2)) # both right aligh

print('{:<5d} {:>7d}'.format(1, 2)) # left, right

print('{:>5d} {:<7d}'.format(1, 2)) # right, left
