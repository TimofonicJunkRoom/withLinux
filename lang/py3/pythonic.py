#!/usr/bin/env python3

'''
https://www.zhihu.com/question/48755767#answer-47628816
https://www.zhihu.com/question/23760468#answer-5661732
https://stackoverflow.com/questions/101268/hidden-features-of-python#101276
https://www.zhihu.com/question/57470958#answer-56901848
'''

                                                            # problem example 1
a = 4.2
b = 2.1
print((a+b)==6.3) # False

                                                                       # tips 1
L = [ i*i for i in range(5) ]
for idx,value in enumerate(L, 1): # index starts from 1
    print(idx, ':', value)

                                                                       # tips 2

# for item in L[::-1]:
for item in reversed(L):
    print(item)

                                                                       # tips 3
# for row in rows:
#   if row[1]==0 and row[9] != 'YES':
#     return True
# return False
# -> return any(row[1]==0 and row[9] != 'YES' for row in rows)

                                                                       # tips 4
# raise SystemExit('It failed')

                                                                       # tips 5
# if not os.path.exists('myfile'):
#   with open('myfile', 'wt') as f:
#     f.write('content\n')
# else:
#   print('file already exists')
# -> with open('myfile', 'xt') as f:
#      f.write('content\n')

                                                                       # tips 6
# port = kwargs.get('port')
# if port is None:
#   port = 3306
# -> port = kwargs.get('port', 3306)

# -> last = L.pop()
                                                                       # tips 7
#d = {}
#for key,value in pairs:
#    if key not in d:
#        d[key] = []
#    d[key].append(value)
# ->
#d = defaultdict(list)
#for key,value in pairs:
#    d[key].append(value)

#d = defaultdict(int)
#for line in file:
#    for word in line.strip().split():
#        d[word] += 1
# See also: collections.Counter
#word_count = Counter()
#for line in file:
#  word_count.update(line.strip().split())

#result = sorted(zip(d.values(), d.keys(), reverse=True))[:3]
#for val,key in result: print(key, val)
# ->
# for key,val in word_count.most_common(3): print(key,val)

                                                                       # tips 8
#namedtuple

                                                                       # tips 9
#from threading import Thread
#import time
#import random
#from queue import Queue
#
#queue = Queue(10)
#
#class ProducerThread(Thread):
#    def run(self):
#        nums = range(5)
#        global queue
#        while True:
#            num = random.choice(nums)
#            queue.put(num)
#            print('produced', num)
#            time.sleep(random.random())
#
#class ConsumerThread(Thread):
#    def run(self):
#        global queue
#        while True:
#            num = queue.get()
#            queue.task_done()
#            print('consumed', num)
#            time.sleep(random.random())
#
#ProducerThread().start()
#ConsumerThread().start()

# from multiprocessing import Pool
# from multiprocessing.dummy import Pool

                                                                      # tips 10
# decorator: TODO

                                                                      # tips 11
# simple factory
class Shape(object): pass
class Circle(Shape): pass
class Square(Shape): pass

for name in ('Circle', 'Square'):
    cls = globals()[name]
    obj = cls()

                                                                     # tips 2/1
class ObjectDict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)
    def __setattr__(self, name, value):
        self[name] = value

                                                                     # tips 3/1
# context manager
class OpenContext(object):
    def __init__(self, filename, mode):
        self.fp = open(filename, mode)
    def __enter__(self):
        return self.fp
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.fp.close()
# with OpenContext('/tmp/a', 'a') as f:
#   f.write('hello')

                                                                     # tips 4/1
# save memory with __slots__
class Foo(object):
    __slots__ = ['id', 'caption', 'url']
    '''
    use __slots__ when the attributes of the class is fixed.
    Then python will not use a dict to store the attributes, but with a list.
    '''
    def __init__(self, id, caption, url):
        self.id = id
        self.caption = caption
        self.url = url

                                                                     # misc
a = 1
b = 4
c = [b,a][a>b]

import random
a = [random.randint(0,9) for _ in range(10)]
a.sort(key=lambda x:x)
a = [(random.randint(0,9), random.random()) for _ in range(20)]
a.sort(key=lambda x: (x[0], x[1]))

class Test:
    def __init__(self):
        for i in range(10):
            exec("self.p{} = lambda: print({})".format(i, i))
        for i in range(10): exec("self.p{}".format(i))
x = Test()
x.p0()
x.p9()

list_1 = [[1, 2], [3, 4, 5], [6, 7], [8], [9]]
list_2 = []
for x in list_1: list_2 += x
list_2 = [elem for group in list_1 for elem in group]
sum(list_1, [])

from functools import reduce
reduce(lambda x,y: x+y, list_1)

for i in [1,1,2]:
    if i == 0: break
else:
    print('i \\neq 0 \\forall i')
for i in [1,1,0]:
    if i == 0: break
else:
    print('i \\neq 0 \\forall i')

4<6<8
4<6<1
