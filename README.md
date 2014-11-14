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
So, likely that parallel computing doesn't help I/O intensive tasks much.  
Demo:
```
$ ./bytefreq -l bytefreq.c
Crunching data ...
=========================================================
Character    Count           of_ALL          of_Specified
=========    ============    ============    ============
(0x61, a)             243         3.213 %         6.080 %
(0x62, b)              66         0.873 %         1.651 %
(0x63, c)             234         3.094 %         5.854 %
(0x64, d)             105         1.389 %         2.627 %
(0x65, e)             400         5.290 %        10.008 %
(0x66, f)             124         1.640 %         3.102 %
(0x67, g)              47         0.622 %         1.176 %
(0x68, h)              82         1.084 %         2.052 %
(0x69, i)             204         2.698 %         5.104 %
(0x6a, j)               0         0.000 %         0.000 %
(0x6b, k)              53         0.701 %         1.326 %
(0x6c, l)             199         2.632 %         4.979 %
(0x6d, m)             143         1.891 %         3.578 %
(0x6e, n)             318         4.205 %         7.956 %
(0x6f, o)             361         4.774 %         9.032 %
(0x70, p)             142         1.878 %         3.553 %
(0x71, q)              12         0.159 %         0.300 %
(0x72, r)             320         4.232 %         8.006 %
(0x73, s)             184         2.433 %         4.603 %
(0x74, t)             389         5.144 %         9.732 %
(0x75, u)             182         2.407 %         4.553 %
(0x76, v)              30         0.397 %         0.751 %
(0x77, w)              21         0.278 %         0.525 %
(0x78, x)              87         1.150 %         2.177 %
(0x79, y)              48         0.635 %         1.201 %
(0x7a, z)               3         0.040 %         0.075 %
Maximous of specified : (0x65  e) : 400
Minimous of specified : (0x6A, j) : 0
Total specified : 3997, 52.856%
Total read()    : 7562

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
