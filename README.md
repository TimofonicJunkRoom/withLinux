# cda - cd into Archive (tarball)
[unix,c] cd into archive (tarball)

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
* created temp dir "/tmp/cda.CRllN2"
* p: fork() [4844]
* detected [ .tar.gz | .tgz ]
* child terminated (0).
* step into tempdir /tmp/cda.CRllN2
* now pwd = /tmp/cda.CRllN2
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
exit
* cda: OK, removing temp directory "/tmp/cda.CRllN2"...
* fork() [4861]
* child terminated (0).
$ 
```
Note that:
* For safety, by default cda really invoke `rm -i -rf DIR` to remove temp dir.
  If you want to remove it directly, run `$ cda ARCHIVE -f`

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
