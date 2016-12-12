Introducing Python
===
> Oreilly, Bill Lubanovic  

# chapter2: py ingredients: numbers, strings, and variables

int division and float division
```
In [2]: 9/5
Out[2]: 1.8

In [3]: 9//5
Out[3]: 1
```

# chapter2: py filling: lists, tuples, dictionaries, and sets

list copying
```
a = [ 1, 2, 3 ]
b = a # surprise
c = a.copy()
d = list(a)
e = a[:]
```

```
a = [ ... ]
b = ...
c = ...
if len(set(map(len, [a, b, c]))) != 1:
  raise Expception
```

# chapter 4: py crust: code structures

Iterate multiple sequences with `zip()`.

function as closure
```
def a(string):
  def b():
    return "... %s" % string
  return b
```

generators
```
def my_range(first=0, last=10, step=1):
  number = first
  while number < last:
    yield number
    number += step

my_range # function
ranger = my_range(1, 5) # ranger is a generator
for x in ranger: # traversal
```

# chapter 5: py boxes: modules, packages, and programs

python standard library
```
handle missing keys with setdefault() and defaultdict()

count items with Counter()
  from collections import Counter
  breakfast = [ 'spam', 'eggs' , 'spam', 'spam' ]
  breakfast_counter = Counter(breakfast) -> Counter({ 'spam': 3, 'egg': 1 })

order by key with OrderedDict()
  from collections import OrderedDict

stack + queue = deque
  from collections import deque

  e.g.
  def palindrome(word):
    from collections import deque
    dq = deque(word):
    while len(dq) > 1:
      if dq.popleft() != dq.pop():
        return False
    return True

print nicely with pprint()
  from pprint import pprint
```

# chapter 6: objects and classes

get help fron the parent class with super
```
class Person():
  def __init__(self, name):
    self.name = name

class EmailPerson(Person):
  def __init__(self, name, email):
    super().__init__(name) # super() -> Person
    self.email = email
```

method types
```
class A():
  count = 0
  def __init__(self):
    A.count += 1
  def exclaim(self):
    print("I'm an A!")
  @classmethod
  def kids(cls):
    print(cls.count, ' A objects found')
  @staticmethod
  def commercial():
    print("hello")

A.commercial() # ok
a = A()
b = A()
A.kids() # 2
```

page 135
