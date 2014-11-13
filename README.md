Bytefreq
========
UNDER DEVELOPMENT  

[C,util, UNIX-Like] Byte/Char Frequency, Serial/Parallel.  

[bytefreq.c](./bytefreq.c)  
The Main Bytefreq utility, it supports both serial and parallel count approaches,
	and the performance of them need to be tested.  
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
#### compile
just make.
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
For example, $ tr -s 'abc' 'xyz'  
abcdefxyz		<- from stdin  
xyzdefxyz		<- processed by tr  
```  

### Licence
The MIT licence.  
