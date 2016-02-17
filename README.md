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
Synopsis:
	cda [options] ARCHIVE
Description:
	Extract the specified archive into a temporary directory,
	where a shell will be opened for you. This temporary
	directory will be removed in the exitting of shell.
Options:
	-d <DIR>  Specify the temp directory to use.
			(would override the CDA env).
	-l        Also list archive components.
	-L        Only list archive components.
	-X        Only extract the archive.
Environment:
	CDA       Set temp dir to use.  (current: /tmp)
	CDASH     Set shell to use.     (current: /bin/bash)
Version:
	CDA 1.2  <-  libarchive 3.1.2
```

## Example 
```
$ cda /tmp/a.tar.gz
I0117 02:46:54.985 07871 cda.c:147] entering into archive [/tmp/a.tar.gz]
I0117 02:46:54.985 07871 cda.c:157] access("/tmp/a.tar.gz", R_OK) success.
I0117 02:46:54.986 07871 cda.c:178] create temporary directory [/tmp//cda.t2BcJ2]
I0117 02:46:55.016 07871 cda.c:196] libarchive operations are successful. (0).
I0117 02:46:55.016 07871 cda.c:202] fork and execve a shell for you, under [/tmp/cda.t2BcJ2]
I0117 02:46:55.016 07871 cda.c:203] 
W0117 02:46:55.016 07871 cda.c:204] -*- Please exit this shell when your operation is done -*-
I0117 02:46:55.016 07871 cda.c:205] 

$ ls -l
total 4
drwxr-xr-x 3 lumin lumin 4096 Feb 17 02:37 cda
$ exit
exit

I0117 02:46:58.498 07871 cda.c:221] removing the temporary directory [/tmp/cda.t2BcJ2]
W0117 02:46:58.499 07889 cda.c:338]  execve(): rm -rf /tmp/cda.t2BcJ2 
I0117 02:46:58.518 07871 cda.c:346] removal status on [/tmp/cda.t2BcJ2] (0) - Success.
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
