foo
===
Brute force MD5 cracker.  
It can only crack 4-char string, 
but it can be extended/rewrited to fit other lengths of strings.  

###Usage/Example
* Generate a raw MD5 digest file  
```
$ ./md5bin 0000 > 0000.md5
```
here I calculates the MD5 of "0000", then put it in file "0000.md5"  

* Modify foo.c according to your need, such as  
```
- #define MD5_FILE_TO_CRACK "hhhh.md5"
+ #define MD5_FILE_TO_CRACK "0000.md5"
```

* then make
```
$ make
```

* run and wait for answer, you can measure the time meanwhile.
```
$ time ./foo
```
after some seconds, it will throw this stdout:
```
0000
```
