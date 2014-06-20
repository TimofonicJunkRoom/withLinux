## a8freq
---
Simply show freqency of alphabets in file or stream.

This is just a *very simple linux c program* and, so it tends to be something to play around with.    
Just as the brief descriiption above, you can crack certain type of *classic codes* with it.  
  
Usage:
```
a8freq [-hs]Usage : ./a8freq [-hs] [FILE]
Show the alphabets' freqency in file.
If FILE is not specified, stdin would be used.
  -h    Print this help message
  -p	(not implemented)set decimal places in output
  -s    Use another output format
```
  
  
For example,
```
$ a8freq a8freq.c
```
will give this output:
```
A     92	 5.73208723% 
B	 27	 1.68224299% 
C	 83	 5.17133956% 
D	 55	 3.42679128% 
E	 172	 10.71651090% 
F	 81	 5.04672897% 
G	 34	 2.11838006% 
H	 42	 2.61682243% 
I	 131	 8.16199377% 
J	 7	 0.43613707% 
K	 6	 0.37383178% 
L	 77	 4.79750779% 
M	 30	 1.86915888% 
N	 133	 8.28660436% 
O	 92	 5.73208723% 
P	 58	 3.61370717% 
Q	 18	 1.12149533% 
R	 120	 7.47663551% 
S	 84	 5.23364486% 
T	 152	 9.47040498% 
U	 80	 4.98442368% 
V	 8	 0.49844237% 
W	 8	 0.49844237% 
X	 4	 0.24922118% 
Y	 7	 0.43613707% 
Z	 4	 0.24922118% 
ALL 1605 alphabets.
```
  
---
Concerning classic code see http://en.wikipedia.org/wiki/Cryptology   
and http://en.wikipedia.org/wiki/Substitution_cipher  

