CXX ?= clang++
INCLUDES = -I/usr/include/hdf5/serial
LIBS = -lhdf5_cpp -lhdf5_hl_cpp -lhdf5_serial
CXXFLAGS += -std=c++11 -Wall -g -O2 -fopenmp -march=native -DUSE_OPENMP #-fno-inline-functions -Og

test: dataloader tensor blob layer

demo.h5:
	python3 gen-demoh5.py

.PHONY: dataloader tensor blob layer
dataloader: demo.h5
	$(CXX) $(INCLUDES) $(LIBS) $(CXXFLAGS) -o test_dataloader.elf -DLITE_TEST_DATALOADER dataloader.cc 
	valgrind --leak-check=yes ./test_dataloader.elf #OK
tensor: demo.h5
	$(CXX) $(INCLUDES) $(LIBS) $(CXXFLAGS) -o test_tensor.elf -DLITE_TEST_TENSOR tensor.cc 
	valgrind --leak-check=yes ./test_tensor.elf #OK
blob: demo.h5
	$(CXX) $(INCLUDES) $(LIBS) $(CXXFLAGS) -o test_blob.elf -DLITE_TEST_BLOB blob.cc 
	valgrind --leak-check=yes ./test_blob.elf #OK
layer: demo.h5
	$(CXX) $(INCLUDES) $(LIBS) $(CXXFLAGS) -o test_layer.elf -DLITE_TEST_LAYER layer.cc 
	valgrind --leak-check=yes ./test_layer.elf #OK, accuracy layer problematic

.PHONY: test_lineq test_mnist_reg test_toycls
test_lineq:
	$(CXX) $(INCLUDES) $(LIBS) $(CXXFLAGS) -o test_lineq.elf test_lineq.cc
	valgrind --leak-check=yes ./test_lineq.elf #OK
test_mnist_reg:
	$(CXX) $(INCLUDES) $(LIBS) $(CXXFLAGS) -o test_mnist_reg.elf test_mnist_reg.cc
	valgrind --leak-check=yes ./test_mnist_reg.elf #OK
test_toycls:
	$(CXX) $(INCLUDES) $(LIBS) $(CXXFLAGS) -o test_toycls.elf test_toycls.cc 
	valgrind --leak-check=yes ./test_toycls.elf #OK
test_mnist_cls:
	$(CXX) $(INCLUDES) $(LIBS) $(CXXFLAGS) -o test_mnist_cls.elf test_mnist_cls.cc
	valgrind --leak-check=yes ./test_mnist_cls.elf #OK

clean:
	-$(RM) demo.h5 *.elf
