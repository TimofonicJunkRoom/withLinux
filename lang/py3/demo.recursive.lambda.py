(lambda v: 0 if len(v)<1 else v[0] if len(v)==1 else v[0]+s(v[1:]))([1,2,3,4])
