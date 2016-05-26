#!/usr/bin/python3
# 2016 lumin, simple parallelization in python3

import time
import math

class Timer:
  def __init__(self):
    self.reset()
  def reset(self):
    self.start = 0.0
    self.end = 0.0
    self.interval = 0.0
  def set(self):
    self.start = time.time()
  def stop(self):
    self.end = time.time()
    self.interval = self.end - self.start
    print('costs time %f s' % self.interval)

timer = Timer()
task = range(20000000)

print('I: using for-loop')
timer.set()
result1 = []
for i in task:
  result1.append(math.sin(i))
timer.stop()
#print(result1)

print('I: using map()')
timer.set()
result2 = map(math.sin, task)
timer.stop()
#print(result2)


print('I: using ThreadPool')
from multiprocessing.dummy import Pool as ThreadPool

timer.set()
pool = ThreadPool()
result3 = pool.map(math.sin, task)
pool.close()
pool.join()
timer.stop()

print('I: using Pool')
from multiprocessing import Pool

timer.set()
pool2 = Pool()
result4 = pool2.map(math.sin, task)
pool2.close()
pool2.join()
timer.stop()

print('I: this technique is awesome!')
'''
def convert_image(imagepath):
  subprocess.call(['convert', '-resize', '64x64',
    imagepath, imagepath + '_'])

task = [ str(i)+'.jpg' for i in range(1, 2000+1)]
'''
