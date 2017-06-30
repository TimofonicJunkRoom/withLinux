
a = list(map(int,'4 8 7 5 3 3 7 9 6 4 8 1'.split()))

def qsort(v):
    ''' v must be a list '''
    return v if len(v)==0 else \
            (qsort([x for x in v[1:] if x>=v[0]]) + \
            [v[0]] + \
            qsort([x for x in v[1:] if x<v[0]]))

print(a)
print(qsort(a))
