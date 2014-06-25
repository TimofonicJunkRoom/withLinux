# a8freq : a(lphabets) freqency.
Simple linux c tools, for generating and cracking a certain type of substitution cipher.  
This is just a set of very simple and small linux tools.  
  
New features are being added.  
This is free software, at the same time the author account it as a part of learning C and playing.  
  
[a8freq.c]() Simply show freqency of alphabets in file.  
[a8lu.c]() Convert alphabets between upper and lower case.  
[a8shift.c]() Shift alphabets by (+/-)N positions. 
[a8swap.c]() (upcoming) swap pairs of alphabets in file. 
  
For their Usage, look up the c file, and there is function "Usage ()" in each .c file.  
  
### The purpose
According to some Theories, certain types of classic codes, such as substitution cipher, can be cracked via alphabet freqency analyze.  

see http://en.wikipedia.org/wiki/Cryptology  
see http://en.wikipedia.org/wiki/Substitution_cipher  

Have fun play with it ! :-)  
  
  
---
### Examples

#### Generate Substitution cipher
There is an rough and wild way that works:  
```
$ ORIGIN='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
# for example shift them by 1 position.
$ SUBSTI='BCDEFGHIJKLMNOPQRSTUVWXYZAbcdefghijklmnopqrstuvwxyza'
  
$ cat FILE | tr -s $ORIGIN $SUBSTI
```
If you want to simply shift them by (int)N positions, use [a8shift.c]
```
$ a8shift -o 1
# will get same result as above
```

#### a8freq
```
$ a8freq a8freq.c
```
will give this output:
```
A 	 92		 5.73208723% 
B	 27		 1.68224299% 
C	 83		 5.17133956% 
D	 55		 3.42679128% 
E	 172		 10.71651090% 
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
Z	 4		 0.24922118% 
ALL 1605 alphabets.
```
