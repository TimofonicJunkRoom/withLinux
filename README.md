# cda - change directory into Archive

### SYNOPSIS
`cda [options] <ARCHIVE> `  
`cda.sh <ARCHIVE>`

### DESCRIPTION
`cda` is a command line utility that helps you "enter into" or "chdir() into" an archive conveniently.  
You can consider it as an Enhaced version of `cd` command in shell. It builds on [libarchive](https://github.com/libarchive/libarchive) so that it supports many types of archives.
  
`cda.sh` is a archivemount wrapper which implements cda in shell.  

The idea of cda is:  
1. create a temporary directory  
2. extract the archive specified in command line into the temporary directory  
3. `fork()` and then `execve()` a shell in the temporary directory  
4. when the shell is quited, cda deletes the temporary directory recursively  
Hence "change directory into archive" can be implemented.  

```shell
Usage:
  ./cda [options] ARCHIVE
Options:
  -d <DIR>  Specify the temp directory to use.
            (would override the CDA env).
  -f        Force remove tmpdir, instead of interactive rm.
  -l        Also list archive components.
  -L        Only list archive components.
  -X        Only extract the archive.
Environment:
  CDA       Set temp dir to use.  (current: /tmp)
  CDASH     Set shell to use.     (current: /bin/bash)

Dependency    : libarchive 3.1.2
CDA Version   : 1.0~rc1
```

## Example 
```
$ CDASH=/bin/sh cda a.tar.gz
I: CDASH = "/bin/sh"
I: processing archive "a.tar.gz"
I: Create temporary directory [/tmp//./cda.gpsq5F]
I: Child archive handler done. (0).
I: working at destdir [/tmp/cda.gpsq5F]
I: Please exit this shell when your operation is done.

$ ls -l
total 4
drwxr-xr-x 3 lumin lumin 4096 Feb 15 07:03 cda
$ exit

I: remove temp directory [/tmp/cda.gpsq5F]
I: remove the temporary directory [/tmp/cda.gpsq5F] (0) - Success.
```

## Compile & install
* install dependency: `sudo apt install libarchive-dev`  
* compile: `make`  
* install: `sudo make install`  

##LICENSE
```
GPL-3
COPYRIGHT (C) 2016 Lumin Zhou
```
