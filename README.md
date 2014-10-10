fakehttpd
=========
! Under development.  
! It can serve some simple-and-single .html files,  
! but many more detailes are to be finished.  
 
## Files
[fakehttpd.c](fakehttpd.c)  
This is a very tiny and single-file httpd,  
I say it is "fake" because it doses not stick to  
HTTP/1.0 or HTTP/1.1 protocol tightly.  
  
[index.html](index.html)  
An empty html for testing the server.  

## Brief Usage
* Compile :  
```
$ make
```
* Launch the Server :  
```
$ ./fakehttpd some_web_pages.html
```
* For more detail :  
```
$ ./fakehttpd -h
```

## Design
* It can only pass 1 file at most  
* MUCH LESS functions than a norman httpd  

## Case of Use
* Open a single 'STATIC HTML PAGE' to the public  
* Open a single FILE to public (still doing)  

## BUG & ISSUE
* It will crash the web browser if misused   
* Logging function missed  

##Licence
The MIT Licence.
