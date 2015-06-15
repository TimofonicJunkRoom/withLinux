# cda - cd into Archive(tarball)
[unix,c] cd into archive (tarball)

## what this ?
Initially I have a tarball
```shell
$ ls -l test.tar.gz 
-rw-r--r-- 1 lumin lumin 186 Jun 15 16:01 test.tar.gz
```
Then I invoke this "cda" in order to "chdir()" into this archive
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
./test/b
./test/c
./test/d
./test/e
./test/f
```

## Compile & install
* compile: `make`
* install: `make install`

## Hints
* you can set the variable `debug` to 0 in `cda.c` to hide debug info.
* cda invoke RM to remove temporary directory, and now `RM = "echo"`

##LICENSE
`GPL-3+`
