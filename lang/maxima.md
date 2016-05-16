Maxima Note
===
> The Maxima Book  
> Paulo Ney de Souza  
> Richard J. Fateman  
> Joel Moses  
> Cliff Yapp  
> 19th September 2004  

## Chap3, the basics

basic operations `+ - * /`
```
(%i2) 9/4;
                                       9
(%o2)                                  -
                                       4
```

quit
```
quit();
```

end of entry
```
(%i1) x^5+3*x^4+2*x^3+5*x^2+4*x+7;
                        5      4      3      2
(%o1)                  x  + 3 x  + 2 x  + 5 x  + 4 x + 7
```

labels
```
x + 1;
solve(%o1 =0, x);

(%i9) x^2+x+1;
                                   2
(%o9)                             x  + x + 1
(%i10) solve(%o9=0, x);
                         sqrt(3) %i + 1      sqrt(3) %i - 1
(%o10)            [x = - --------------, x = --------------]
                               2                   2
```

custom label
```
(%i11) myeq:x^2+4*x+3;
                                  2
(%o11)                           x  + 4 x + 3
(%i13) solve(myeq=0, x);
(%o13)                        [x = - 3, x = - 1]
```

to or not to evaluate
```
(%i10) diff(1/sqrt(1+x^3),x);
(%i11) ’diff(1/sqrt(1+x^3),x);
```

environment
```
> ev(solve(a*x^2+b*x+c=d,x),a=3,b=4,c=5,d=6);
> a; --> null
```

how high a power you like maxima to expand
```
ev((x+y)^5+(x+y)^4+(x+y)^3+(x+y)^2+(x+y)+(x+y)^-1+(x+y)^-2+(x+y)^-3+(x+y)^-4+(x+y)^-5,EXPAND(3,3));
```

numerical output
```
a:9/4;
exp(a);
ev(exp(a),FLOAT);
ev(exp(a*x),FLOAT);
numerval(b, 25);
a*b;
ev(a*b, FLOAT);
ev(a*b, NUMBER);
float(a);
float(b);
float(a*b);
```

local value
```
eqn1:’diff(x/(x+y)+y/(y+z)+z/(z+x),x);
ev(eqn1,diff);
ev(eqn1,y=x+z);
ev(eqn1,y=x+z,diff);

eqn4:f(x,y)*’diff(g(x,y),x);
ev(eqn4,f(x,y)=x+y,g(x,y)=x^2+y^2);
ev(eqn4,f(x,y)=x+y,g(x,y)=x^2+y^2,DIFF);

eqn1:f(x,y)*’diff(g(x,y),x);
eqn2:3*y^2+5*y+7;
ev(eqn1,g(x,y)=x^2+y^2,f(x,y)=5*x+y^3,solve(eqn2=5,y));
ev(eqn1,g(x,y)=x^2+y^2,f(x,y)=5*x+y^3,solve(eqn2=1,y),diff);
ev(eqn1,g(x,y)=x^2+y^2,f(x,y)=5*x+y^3,solve(eqn2=1,y),diff,FLOAT);
```

TODO page 37
