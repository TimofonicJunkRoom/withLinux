# cda - cd into Archive (tarball)
[unix,c] cd into archive (tarball)

## what's this ?
Assume that I have a tarball
```shell
$ ls -l test.tar.gz 
-rw-r--r-- 1 lumin lumin 186 Jun 15 16:01 test.tar.gz
```
Then I invoke this "cda" in order to "chdir()" into target archive
```shell
$ cda test.tar.gz 
* Extract Archive "test.tar.gz"
* detected [ .tar.gz | .tgz ]
* p: child terminated.
* step into tempdir /tmp/cda.b5WBt3
```
Now we are "in" the archive:
```shell
$ find
.
./test
./test/a
[...]
```
Then exit the shell
```
$ exit
exit
* [OK] now removing temp directory
$ 
```
Note that:
* For safety, currently cda don't really invoke `rm` to remove temp dir.

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
