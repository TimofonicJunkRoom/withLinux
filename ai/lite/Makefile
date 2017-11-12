CXX ?= clang++
INCLUDES = -I/usr/include/hdf5/serial
LIBS = -lhdf5_cpp -lhdf5_hl_cpp -lhdf5_serial
CXXFLAGS += -std=c++11 -Wall -g -O2 -fopenmp -march=native -DUSE_OPENMP #-fno-inline-functions -Og

test: dataloader tensor blob layer

demo.h5:
	python3 gen-demoh5.py

# -- Basic Unit Test
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

# -- High level Tests
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

# -- Benchmarks
.PHONY: benchmark
BASEFLAG= -std=c++11 -Wall -fopenmp -DUSE_OPENMP
benchmark:
	# generate the fake dataset first
	python3 gen-demoh5.py mnist
	# change the code to use the fake dataset, and change iterations
	cp test_mnist_cls.cc test_benchmark.cc
	sed -i -e 's/mnist.h5/mnist.fake/g' test_benchmark.cc

	# compile with -O0 and test
	$(CXX) $(INCLUDES) $(LIBS) $(BASEFLAG) -O0 -o benchmark.elf test_benchmark.cc
	sudo perf stat ./benchmark.elf > /dev/null

	# compile with -O0 -march=native and test
	$(CXX) $(INCLUDES) $(LIBS) $(BASEFLAG) -O0 -march=native -o benchmark.elf test_benchmark.cc
	sudo perf stat ./benchmark.elf > /dev/null

	# compile with -O0 -march=native -flto and test
	$(CXX) $(INCLUDES) $(LIBS) $(BASEFLAG) -O0 -march=native -flto -o benchmark.elf test_benchmark.cc
	sudo perf stat ./benchmark.elf > /dev/null

	# compile with -O2 and test
	$(CXX) $(INCLUDES) $(LIBS) $(BASEFLAG) -O2 -o benchmark.elf test_benchmark.cc
	sudo perf stat ./benchmark.elf > /dev/null

	# compile with -O2 -march=native and test
	$(CXX) $(INCLUDES) $(LIBS) $(BASEFLAG) -O2 -march=native -o benchmark.elf test_benchmark.cc
	sudo perf stat ./benchmark.elf > /dev/null

	# compile with -O2 -flto and test
	$(CXX) $(INCLUDES) $(LIBS) $(BASEFLAG) -O2 -flto -o benchmark.elf test_benchmark.cc
	sudo perf stat ./benchmark.elf > /dev/null

	# compile with -O2 -march=native -flto and test
	$(CXX) $(INCLUDES) $(LIBS) $(BASEFLAG) -O2 -march=native -flto -o benchmark.elf test_benchmark.cc
	sudo perf stat ./benchmark.elf > /dev/null

	# compile with -O3 -march=native -flto and test, some times the aggressive optimization may not improve performance.
	$(CXX) $(INCLUDES) $(LIBS) $(BASEFLAG) -O3 -march=native -flto -o benchmark.elf test_benchmark.cc
	sudo perf stat ./benchmark.elf > /dev/null

clean:
	-$(RM) demo.h5 *.elf
