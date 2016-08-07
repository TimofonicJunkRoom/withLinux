main:
	$(CC) -shared -fPIC -g -Wall kernel.c -o kernel.so
	rustc --crate-type dylib kernel.rs
	$(CXX) -shared -fPIC -g -Wall kernel.cc -o cxxkernel.so
python:
	python3 myffi.py
rust:
	python3 ffirust.py
cxx:
clean:
	$(RM) kernel.so libkernel.so cxxkernel.so
