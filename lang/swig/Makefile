main: python3 lua

python3: clean
	swig -python -o ex_wrap.c ex.i
	gcc -c -fpic ex.c ex_wrap.c -I/usr/include/python3.5m
	gcc -shared ex.o ex_wrap.o -o _ex.so
	python3 -c "import ex; print(ex.factorial(10))"

lua: clean
	swig -lua -o ex_wrap.c ex.i 
	gcc -c -fpic ex.c ex_wrap.c -I/usr/include/lua5.1 
	gcc -shared ex.o ex_wrap.o -o ex.so
	luajit -e "require 'ex'; print(ex.factorial(10))"

clean:
	$(RM) ex_wrap.c
	$(RM) -f *.o *.so
	$(RM) -rf __pycache__ *.py
