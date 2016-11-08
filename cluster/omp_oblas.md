Note on OpenMP and OpenBLAS
===

# when running torch
please set openmp thread number according to your CPU or it will suck in performance.
```
OMP_NUM_THREADS=6 th test.lua
```

# openblas
please always re-compile openblas locally to gain better performance.
```
apt source openblas
vim rules
 -> export DEB_CFLAGS_MAINT_APPEND= -march=native -mtune=native # C
 -> export DEB_FFLAGS_MAINT_APPEND= -march=native -mtune=native # Fortran
debuild
```
