CXX=g++
CXXFLAGS=-Wall -Wpedantic -g
main:
	$(CXX) $(CXXFLAGS) test_ds.cc -o test_ds
	./test_ds
dox:
	doxygen
clean:
	-$(RM) -rf doxygen
	-$(RM) test_ds
