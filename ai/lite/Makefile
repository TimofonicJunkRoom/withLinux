CXX ?= clang++
INCLUDES = -I/usr/include/hdf5/serial
LIBS = -lhdf5_cpp -lhdf5_hl_cpp -lhdf5_serial
CXXFLAGS += -Wall -g -O2 -fopenmp -march=native -DUSE_OPENMP #-fno-inline-functions 

test: dataloader tensor layer testnet

demo.h5:
	python3 gen-demoh5.py

dataloader: demo.h5
	$(CXX) $(INCLUDES) $(LIBS) $(CXXFLAGS) -o test_dataloader -DLITE_TEST_DATALOADER dataloader.cc 
	valgrind --leak-check=yes ./test_dataloader #OK
tensor: demo.h5
	$(CXX) $(INCLUDES) $(LIBS) $(CXXFLAGS) -o test_tensor -DLITE_TEST_TENSOR tensor.cc 
	valgrind --leak-check=yes ./test_tensor #OK
blob: demo.h5
	$(CXX) $(INCLUDES) $(LIBS) $(CXXFLAGS) -o test_blob -DLITE_TEST_BLOB blob.cc 
	valgrind --leak-check=yes ./test_blob #OK
layer: demo.h5
	$(CXX) $(INCLUDES) $(LIBS) $(CXXFLAGS) -o test_layer -DLITE_TEST_LAYER layer.cc 
	valgrind --leak-check=yes ./test_layer #issue

test_lineq:
	$(CXX) $(INCLUDES) $(LIBS) $(CXXFLAGS) -o test_lineq.elf test_lineq.cc
	valgrind --leak-check=yes ./test_lineq.elf

testnet_cls: demo.h5
	$(CXX) $(INCLUDES) $(LIBS) $(CXXFLAGS) -o testnet_cls_test testnet_cls.cc 
	./testnet_cls_test
testnet_mnist:
	$(CXX) $(INCLUDES) $(LIBS) $(CXXFLAGS) -o testnet_mnist_test testnet_mnist.cc
	./testnet_mnist_test
testnet_mnist_reg:
	$(CXX) $(INCLUDES) $(LIBS) $(CXXFLAGS) -o testnet_mnist_reg_test testnet_mnist_reg.cc
	./testnet_mnist_reg_test

.PHONY: dataloader tensor blob layer test_lineq

clean:
	-$(RM) dataloader_test demo.h5 tensor_test layer_test testnet_cls_test testnet_reg_test testnet_mnist_test testnet_mnist_reg_test *.elf
