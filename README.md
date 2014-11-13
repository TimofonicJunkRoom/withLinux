Bytefreq
========

[C,util, UNIX-Like] Byte/Char Frequency, Serial/Parallel.  

[bytefreq.c](./bytefreq.c)  
The Main Bytefreq utility, it supports both serial and parallel count approaches,
	and the performance of them need to be tested.  
The parallel appraoch is implemented with OpenMP, with juse one "#pragma" added.  
[util/a8lu.c](./util/a8lu.c)     
Convert alphabets between upper and lower case.   
[util/a8shift.c](./util/a8shift.c)  
Shift alphabets by (+/-)N positions in alphabet list.  
  
---
### Bytefreq
As following said, and additionally, ANSI color is used in the print funtion.  
```
$ bytefreq -h
Usage:
	./bytefreq [options] [FILE]
Description:
	Count the frequency of specified char.
	Only shows Total read size if no char specified.
	If no <FILE> is given, it would count from the stdin.
Options:
	-h show this help message
	-V show version info
	-v verbose mode
	-p use parallel approach
	-d don't use percent output, use float instead
	-A specify all bytes to count
	-l specify lower to count
	-u specify upper to count
	-s specify symbol to count
	-c specify control character to count
	-a specify alphabets to count (= '-lu')
	...
	for more info see -v
```
In fact, the parallel approach (OpenMP) seems to have lower performance than normal Serial one. 
Demo:
```
$ ./bytefreq -l Makefile
Crunching data ...
(0x61, a) : 11 | 4.151% of spec | 2.895% of ALL
(0x62, b) : 17 | 6.415% of spec | 4.474% of ALL
(0x63, c) : 9 | 3.396% of spec | 2.368% of ALL
(0x64, d) : 2 | 0.755% of spec | 0.526% of ALL
(0x65, e) : 32 | 12.075% of spec | 8.421% of ALL
(0x66, f) : 13 | 4.906% of spec | 3.421% of ALL
(0x67, g) : 6 | 2.264% of spec | 1.579% of ALL
(0x68, h) : 6 | 2.264% of spec | 1.579% of ALL
(0x69, i) : 18 | 6.792% of spec | 4.737% of ALL
(0x6a, j) : 0 | 0.000% of spec | 0.000% of ALL
(0x6b, k) : 1 | 0.377% of spec | 0.263% of ALL
(0x6c, l) : 19 | 7.170% of spec | 5.000% of ALL
(0x6d, m) : 8 | 3.019% of spec | 2.105% of ALL
(0x6e, n) : 21 | 7.925% of spec | 5.526% of ALL
(0x6f, o) : 7 | 2.642% of spec | 1.842% of ALL
(0x70, p) : 10 | 3.774% of spec | 2.632% of ALL
(0x71, q) : 9 | 3.396% of spec | 2.368% of ALL
(0x72, r) : 21 | 7.925% of spec | 5.526% of ALL
(0x73, s) : 16 | 6.038% of spec | 4.211% of ALL
(0x74, t) : 19 | 7.170% of spec | 5.000% of ALL
(0x75, u) : 9 | 3.396% of spec | 2.368% of ALL
(0x76, v) : 0 | 0.000% of spec | 0.000% of ALL
(0x77, w) : 1 | 0.377% of spec | 0.263% of ALL
(0x78, x) : 0 | 0.000% of spec | 0.000% of ALL
(0x79, y) : 10 | 3.774% of spec | 2.632% of ALL
(0x7a, z) : 0 | 0.000% of spec | 0.000% of ALL
Maximous of specified : (0x65  e) : 32
Minimous of specified : (0x6A, j) : 0
Total specified : 265, 69.737%
Total   read()  : 380
```
  
---
#### Expample of util/a8lu.c
```
$ ./a8lu
# lower to upper, read from stdin.
ab cd EF  <- stdin
AB CD EF  <- stdout

$ ./a8lu -r
# upper to lower, read from stdin.
AB CD ef  <- stdin
ab cd ef  <- stdout
```

#### Example of util/a8shift.c
```
$ ./a8shift -o 2
# go right by 2 positions.
ab yz  <- stdin
cd ab  <- stdout

$ ./a8shift -o -2
# go left by 2 positions.
ab yz  <- stdin
yz wx  <- stdout
```

#### Example of util/a8shift.c :: Generate Substitution cipher
There is an rough and wild way that works:  
```
$ ORIGIN='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
# for example shift them by 1 position.
$ SUBSTI='BCDEFGHIJKLMNOPQRSTUVWXYZAbcdefghijklmnopqrstuvwxyza'
  
$ cat FILE | tr -s $ORIGIN $SUBSTI
```
If you want to simply shift them by (int)N positions, use [util/a8shift.c](./util/a8shift.c).  
```
$ cat FILE | ./a8shift -o 1
# same result as above
```

#### Example :: swap characters
The gnu's tr is enough to this purpose.  
```
$ tr -s 'ORIGIN_LIST' 'TARGET_LIST'
```
For example,
```
$ tr -s 'abc' 'xyz'  
abcdefxyz		<- from stdin  
xyzdefxyz		<- processed by tr  
```  

### Licence
The MIT licence.  
