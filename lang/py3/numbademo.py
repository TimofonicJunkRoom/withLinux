#!/usr/bin/python3

from numba import jit
from numpy import arange
import time

# jit decorator tells Numba to compile this function.
# The argument types will be inferred by Numba when function is called.
@jit
def sum2d(arr):
    M, N = arr.shape
    result = 0.0
    for i in range(M):
        for j in range(N):
            result += arr[i,j]
    return result

def sum2d_plain(arr):
    M, N = arr.shape
    result = 0.0
    for i in range(M):
        for j in range(N):
            result += arr[i,j]
    return result

a = arange(65536).reshape(256,256)
t1 = time.time()
print(sum2d(a))
t2 = time.time()

t3 = time.time()
print(sum2d_plain(a))
t4 = time.time()

print('+numba: {}; plain: {}'.format(t2-t1, t4-t3))
