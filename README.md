# DebArchive
  
[Tool] Simple Debian Source Syncer  
  
Download all Sources from the Debian Archive mirror site. Different from
debian mirroring scripts like `debmirror`, `DebArchive` syncs only source code.
  
### Usage
  
1. Specify your target mirror for `rsync` in config. e.g.:
```
SRC="xdlinux.info::debian/"
```
  
2. Just `make`
```
$ make
```

### License
```
BSD-2-Clause
```
