## A long list of linux utilities

where `pkg` means `corresponding package in debian`  

APT/DPKG
---
1. `dpkg -L` lists package contents  
1. `dpkg -S` searchs to which package a file belongs  

Archive Utilities
---
1. `lsar` lists archive contents, pkg `unar`  
1. `unar` extracts contents from archive, pkg `unar`
1. `genisoimage` for generating iso files, pkg `genisoimage`  

Terminal
---
1. `uxterm`  

> note, both `Ctrl^LeftClick` and `Ctrl^RightClick` will trigger menu.  
> note, change font with this:  
```
$ cat .Xdefaults
XTerm*faceName: Ubuntu Mono
XTerm*faceSize: 13
XTerm*background: black
XTerm*foreground: green
```
> note, download ubuntu font here `ubuntu/pool/main/u/ubuntu-font-family-sources/`  

System Monitoring
---
1. `conky`  
1. `dstat`  
1. `htop`  
