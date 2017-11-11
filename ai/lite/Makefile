CXX ?= clang++
INCLUDES = -I/usr/include/hdf5/serial
LIBS = -lhdf5_cpp -lhdf5_hl_cpp -lhdf5_serial
CXXFLAGS += -Wall -g -O2 -fopenmp -march=native -DUSE_OPENMP #-fno-inline-functions 

test: dataloader tensor layer testnet

demo.h5:
	python3 gen-demoh5.py
dataloader: demo.h5
	$(CXX) $(INCLUDES) $(LIBS) $(CXXFLAGS) -o dataloader_test -DLITE_TEST_DATALOADER dataloader.cc 
	valgrind --leak-check=yes ./dataloader_test
tensor: demo.h5
	$(CXX) $(INCLUDES) $(LIBS) $(CXXFLAGS) -o tensor_test -DLITE_TEST_TENSOR tensor.cc 
	valgrind --leak-check=yes ./tensor_test
layer: demo.h5
	$(CXX) $(INCLUDES) $(LIBS) $(CXXFLAGS) -o layer_test -DLITE_TEST_LAYER layer.cc 
	./layer_test
testnet_cls: demo.h5
	$(CXX) $(INCLUDES) $(LIBS) $(CXXFLAGS) -o testnet_cls_test testnet_cls.cc 
	./testnet_cls_test
testnet_reg: demo.h5
	$(CXX) $(INCLUDES) $(LIBS) $(CXXFLAGS) -o testnet_reg_test testnet_reg.cc # seems working
	./testnet_reg_test
testnet_mnist:
	$(CXX) $(INCLUDES) $(LIBS) $(CXXFLAGS) -o testnet_mnist_test testnet_mnist.cc
	./testnet_mnist_test
testnet_mnist_reg:
	$(CXX) $(INCLUDES) $(LIBS) $(CXXFLAGS) -o testnet_mnist_reg_test testnet_mnist_reg.cc
	./testnet_mnist_reg_test

clean:
	-$(RM) dataloader_test demo.h5 tensor_test layer_test testnet_cls_test testnet_reg_test testnet_mnist_test testnet_mnist_reg_test
