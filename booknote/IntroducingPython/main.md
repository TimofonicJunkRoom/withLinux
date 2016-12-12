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
```
