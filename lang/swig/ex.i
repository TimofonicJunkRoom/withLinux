// swig interface file.

%module ex
%{
extern double My_variable;
extern int factorial(int);
extern int my_mod(int n, int m);
%}

extern double My_variable;
extern int factorial(int);
extern int my_mod(int n, int m);

