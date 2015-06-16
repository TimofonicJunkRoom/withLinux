# cda - cd into Archive
[unix,c] cd into archive  
not only tarballs, .7z and .zip are also supported.

## what's this ?
Assume that I have a tarball
```shell
$ ls -l coreutils_8.23.orig.tar.gz 
-rw-r--r-- 1 root root 12582141 Sep  1  2014 coreutils_8.23.orig.tar.gz
```
Then I invoke this "cda" in order to "chdir()" into target archive
```shell
$ cda coreutils_8.23.orig.tar.gz -f
* Extract Archive "coreutils_8.23.orig.tar.gz"
* Created temp dir "/tmp/cda.XoyzAa"
* detected [ .tar.gz | .tgz ]
* Child TAR terminated (0).
* Stepping into Archive (tempdir): /tmp/cda.XoyzAa
* cda: PWD = /tmp/cda.XoyzAa
*      fork and execve bash ...
$ 
```
Now we are "in" the archive:
```shell
$ find | head
.
./coreutils-8.23
./coreutils-8.23/THANKS
[...]
```
Then exit the shell
```
$ exit
* cda: OK, Removing temp directory "/tmp/cda.XoyzAa"...
* Child RM terminated with (0) - Success.
$ 
```
Note that:
* For safety, by default cda really invoke `rm -i -rf DIR` to remove temp dir.
If you want to remove it directly, run `$ cda ARCHIVE -f`

## Supported Archive formats  
* .tar.gz | .tgz
* .tar.bz2 | .tbz | .tbz2
* .tar.xz | .txz
* .tar
* .zip
* .7z
* more in the future

## Without 'cda'
Let's do the same thing as above, withou `cda`
```shell
$ ls -l test.tar.gz
[...]
(1)$ tar zxvf test.tar.gz -C SOME_DIRECTORY
[...]
(2)$ cd SOME_DIRECTORY
......
(3)$ rm -rf SOME_DIRECTORY
```
`cda` encapsules the (1,2,3) steps into one.

## Compile & install
* compile: `make`
* install: `make install`

## Hints
* you can set the variable `debug` to 0 in `cda.c` to hide debug info.

##LICENSE
GPL-3+
