Notable difference between Octave and Matlab
===

* `reshape()`. Suppose we have a textfile
```
0 0 0 1
1 0 0 0
```
then read it
```
data = textread('textfile');
if __MATLAB__
  data = reshape(data, 2, 4);
elseif __OCTAVE__
  data = reshape(data, 4, 2);
endif
```
