CXX ?= clang++
INCLUDES = -I/usr/include/hdf5/serial
LIBS = -lhdf5_cpp -lhdf5_hl_cpp -lhdf5_serial
CXXFLAGS += -Wall -g

test: dataloader tensor layer
demo.h5:
	python3 gen-demoh5.py
dataloader: demo.h5
	$(CXX) $(INCLUDES) $(LIBS) $(CXXFLAGS) -o dataloader_test -DLITE_TEST_DATALOADER dataloader.cc 
	./dataloader_test
tensor: demo.h5
	$(CXX) $(INCLUDES) $(LIBS) $(CXXFLAGS) -o tensor_test -DLITE_TEST_TENSOR tensor.cc 
	./tensor_test
layer: demo.h5
	$(CXX) $(INCLUDES) $(LIBS) $(CXXFLAGS) -o layer_test -DLITE_TEST_LAYER layer.cc 
	./layer_test
clean:
	-$(RM) dataloader_test demo.h5 tensor_test
