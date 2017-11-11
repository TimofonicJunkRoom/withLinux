CXX ?= clang++
INCLUDES = -I/usr/include/hdf5/serial
LIBS = -lhdf5_cpp -lhdf5_hl_cpp -lhdf5_serial
CXXFLAGS += -Wall -g -O2 -fopenmp -march=native -DUSE_OPENMP #-fno-inline-functions 

test: dataloader tensor layer testnet

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
	valgrind --leak-check=yes ./test_layer.elf #issue

.PHONY: test_lineq test_mnist_reg
test_lineq:
	$(CXX) $(INCLUDES) $(LIBS) $(CXXFLAGS) -o test_lineq.elf test_lineq.cc
	valgrind --leak-check=yes ./test_lineq.elf
test_mnist_reg:
	$(CXX) $(INCLUDES) $(LIBS) $(CXXFLAGS) -o test_mnist_reg.elf test_mnist_reg.cc
	valgrind --leak-check=yes ./test_mnist_reg.elf


testnet_cls: demo.h5
	$(CXX) $(INCLUDES) $(LIBS) $(CXXFLAGS) -o testnet_cls_test testnet_cls.cc 
	./testnet_cls_test
testnet_mnist:
	$(CXX) $(INCLUDES) $(LIBS) $(CXXFLAGS) -o testnet_mnist_test testnet_mnist.cc
	./testnet_mnist_test

clean:
	-$(RM) dataloader_test demo.h5 tensor_test layer_test testnet_cls_test testnet_reg_test testnet_mnist_test testnet_mnist_reg_test *.elf
