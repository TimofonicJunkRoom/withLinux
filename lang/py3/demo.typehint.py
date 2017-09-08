#!/usr/bin/python3
'''
typehint requires python3 >= 3.5
https://docs.python.org/3/library/typing.html

You can use the static type checker mypy to check the code
'''

from typing import Any, Callable, Dict, List, Tuple


def foo(idx:int, name:str, scores:List[str]) -> None:
    print(idx, name, scores)
foo('asfd', 'asdf', 'asdf') # works with cpython but `mypy` will warn you.


def scal(alpha:float, vector:List[float]) -> List[float]:
    return [alpha*i for i in vector]
print(scal(1.0, [1., 2., 3.]))


def foobar(f:Callable[[List[int]], List[int]], v:List[int]) -> List[int]:
    return f(v)
def foobarr(f:Callable[[List[int]], int], v:List[int]) -> int:
    return f(v)
def square(v:List[int]) -> List[int]:
    for i,n in enumerate(v):
        v[i] = n*n
    return v
def mysum(v:List[int]) -> int:
    from functools import reduce
    return reduce(lambda a,b: a+b, v)
print(foobar(square, [1, 2, 3]))
print(foobarr(mysum, [1, 2, 3]))


def p(item:Any) -> float:
    from math import e
    print(item)
    return e
p('asdf')


def abcdef(x:Tuple[Tuple[int, int], int]) -> None:
    return
abcdef( ((1,2),3) )
