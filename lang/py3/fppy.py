#!/usr/bin/python3.5
''' functional programming with python 
http://www.oreilly.com/programming/free/files/functional-programming-python.pdf
https://docs.python.org/3.5/howto/functional.html
'''
def fpsum(vec):
    if len(vec)==1:
        return vec[0]
    else:
        return vec[0]+fpsum(vec[1:])

print(fpsum(list(range(101))))
