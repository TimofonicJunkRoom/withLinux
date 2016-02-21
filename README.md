# cda - change directory into Archive

```
$ cd foobar.tar.gz
bash: cd: apt.tgz: Not a directory

$ cda foobar.tar.gz
I0121 14:47:36.618 10035 cda.c:182] @main() Extracting Archive into [/tmp/cda.USfB1J]...
W0121 14:47:37.100 10035 cda.c:203] @main() -*- Exit this shell when operations complete -*-
$ ls -l
total 4
drwxr-xr-x 17 xxx xxx 4096 Jun 11  2015 foobar
```

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
    CDA 1.6.1 (21 Feb. 2016)  <-  libarchive 3.1.2
    built on Feb 21 2016 14:40:07 
```

## Example 

```
$ cda apt.tgz 
I0121 14:50:52.735 10850 cda.c:182] @main() Extracting Archive into [/tmp/cda.CpqmJa]...
W0121 14:50:52.925 10850 cda.c:203] @main() -*- Exit this shell when operations complete -*-
$ ls -l
total 4
drwxr-xr-x 17 lumin lumin 4096 Jun 11  2015 apt
$ exit
exit
I0121 14:51:02.172 10850 cda.c:393] @remove_tmpdir() Removal of [/tmp/cda.CpqmJa] (0) : Success.
$ 
```

## Compile & install
* install dependency: `sudo apt install libarchive-dev`  
* compile: `make` (recommended) or `cmake . ; make`  

##LICENSE
```
GPL-3
COPYRIGHT (C) 2016 Lumin Zhou
```
