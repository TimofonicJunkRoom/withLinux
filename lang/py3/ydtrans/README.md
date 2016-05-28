# ydtrans
[py3] youdao translation shortcut in terminal, very simple one  
  
It translates the argv you'd given, that's all.  

### example  
```
$ python3.4 t.py python
```

### hint
add this into bashrc:
```
t () {
  python3.4 some/dir/t.py $@
}
```
then you can just call the script with
```
$ t <your keyword or sentense>
```
