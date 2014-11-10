Bytefreq
========

[C,util] Byte/Char Frequency, Serial/Parallel.  

THIS PROGRAME IS BEING REWRITED NOW.  

(A set of) Simple linux c tools, for generating and cracking a certain type of substitution cipher.    
  
[Substitution cipher is very easy to be cracked.](http://en.wikipedia.org/wiki/Substitution_cipher)   
According to some Theories, certain types of classic codes, such as substitution cipher, can be cracked via alphabet freqency analyze, see reference below.  
  
New tools, functions, features are being added.     

[bytefreq.c](./bytefreq.c)  
The Main Bytefreq utility, it supports both serial and parallel count method, and is under developing.  
In the future I will delete a8freq and replace it with bytefreq.  
[a8freq.c](./a8freq.c)   
Simply show freqency of alphabets in file.   
Specifying chars to count is supported now (lower, upper, number).  
[a8lu.c](./a8lu.c)     
Convert alphabets between upper and lower case.   
[a8shift.c](./a8shift.c)  
Shift alphabets by (+/-)N positions in alphabet list.  
  
For their Usage, look up the c file, and there is function "Usage ()" in each .c file.  
  
### a8freq :: reference

see http://en.wikipedia.org/wiki/Cryptology  
see http://en.wikipedia.org/wiki/Substitution_cipher  

Have fun playing with it ! :-)  
  
  
---
### Examples

#### Examples :: Generate Substitution cipher
There is an rough and wild way that works:  
```
$ ORIGIN='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
# for example shift them by 1 position.
$ SUBSTI='BCDEFGHIJKLMNOPQRSTUVWXYZAbcdefghijklmnopqrstuvwxyza'
  
$ cat FILE | tr -s $ORIGIN $SUBSTI
```
If you want to simply shift them by (int)N positions, use [a8shift.c](./a8shift.c).  
```
$ cat FILE | ./a8shift -o 1
# same result as above
```

#### Examples :: swap characters
The gnu's tr is enough to this purpose.  
```
$ tr -s 'ORIGIN_LIST' 'TARGET_LIST'
```
For example, $ tr -s 'abc' 'xyz'  
abcdefxyz		<- from stdin  
xyzdefxyz		<- processed by tr  
  

#### Examples :: a8freq
```
$ ./a8freq a8freq.c
# read from file "a8freq.c"
A 	 92		 5.73208723% 
B	 27		 1.68224299% 
C	 83		 5.17133956% 
D	 55		 3.42679128% 
E	 172		 10.71651090%   <- line highlighted in red.
F	 81		 5.04672897%  
G	 34		 2.11838006% 
H	 42		 2.61682243% 
I	 131		 8.16199377% 
J	 7		 0.43613707% 
K	 6		 0.37383178% 
L	 77		 4.79750779% 
M	 30		 1.86915888% 
N	 133		 8.28660436% 
O	 92		 5.73208723% 
P	 58		 3.61370717% 
Q	 18		 1.12149533% 
R	 120		 7.47663551% 
S	 84		 5.23364486% 
T	 152		 9.47040498% 
U	 80		 4.98442368% 
V	 8		 0.49844237% 
W	 8		 0.49844237% 
X	 4		 0.24922118% 
Y	 7		 0.43613707% 
Z	 4		 0.24922118%   <- line highlighted in green. 
ALL 1605 alphabets.
```

#### Expamples :: a8lu
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

#### Examples :: a8shift
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

### Licence
The MIT licence.  
