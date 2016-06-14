Flex Note
===

> package flex, flex-doc on Jessie  
> file:///usr/share/doc/flex-doc/html/index.html  

# File format
```
/* comment */
definitions

  name definition
  %top{
    #include <math.h>
  }

%%

rules

  pattern action

%%

user code

  copied to lex.yy.c verbatim.
```

# Patterns

> file:///usr/share/doc/flex-doc/html/Patterns.html#Patterns  

```
'x'            char x
'.'            any byte except \n
'[xyz]'        either x or y or z
'[a-f]'        match a to f
'[^A-Z]'       except uppercase letter
'[a-z]{-}[aeiou]'
               the lowercase consonants
'r*'           zero or more r, where r can be expr
'r+'           one or more r
'r?'           zero or one r
'r{2,5}'       two to five r's
'r{2,}'        two or more r's
'r{4}'         exactly four r's
'{name}'       expansion of 'name' definition
'\0'           ascii nul
'r|s'          either r or s
'r/s'          r only if followed by s
'^r'           r at the beginning of line
'r$'           r at the end of line
'<<EOF>>'      eof
```

# Generated scanner `yylex()`

```
int yylex() { ... }

/* it reads `yyin`, which is stdin by default */
```



