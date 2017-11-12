CXX = clang++-4.0
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
	@echo
	# you can use "time" instead of "perf"
	
	# warm up and report nothing, the kernel may cache something
	$(CXX) $(INCLUDES) $(LIBS) $(BASEFLAG) -O0 -o benchmark.elf test_benchmark.cc
	./benchmark.elf > /dev/null
	@echo

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

# Here are some results (one-shot, not averaged, unit=second)
#
# [ I5-2520M, g++ 7.2.0-14 Debian (Thinkpad T420s) ]
#
#  -O0                                = 13.25
#  -O0 -march=native                  = 13.01
#  -O0 -march=native -flto            = 13.12
#  -O2                                = 1.81
#  -O2 -march=native                  = 1.75
#  -O2 -flto                          = 1.81
#  -O2 -march=native -flto            = 1.73      (7.65x to -O0)
#  -O3 -march=native -flto            = 1.81
#
# [ I5-2520M, clang++ 4.0.1-8 Debian (Thinkpad T420s) ]
#
#  -O0                                = 14.77
#  -O0 -march=native                  = 14.97
#  -O0 -march=native -flto            = 15.31
#  -O2                                = 1.56
#  -O2 -march=native                  = 1.47
#  -O2 -flto                          = 1.50
#  -O2 -march=native -flto            = 1.52
#  -O3 -march=native -flto            = 1.44      (10.25x to -O0)
#
# [ I7-6900K, g++ 6.4.0-3 Debian ]
#
#  -O0                                = 5.07
#  -O0 -march=native                  = 4.70
#  -O0 -march=native -flto            = 5.02
#  -O2                                = 1.32
#  -O2 -march=native                  = 1.31
#  -O2 -flto                          = 1.35
#  -O2 -march=native -flto            = 1.29      (3.93x to -O0)
#  -O3 -march=native -flto            = 1.38
#
#  OMP_NUM_THREADS=16 : fastest 1.30, slowest 4.96
#  OMP_NUM_THREADS=8  : fastest 1.31, slowest 5.10
#  OMP_NUM_THREADS=4  : fastest 1.21, slowest 5.10 <- Tuning works?
#  OMP_NUM_THREADS=2  : fastest 1.35, slowest 5.00
#
# [ I7-6900K, clang++ 4.0.1-1 Debian ]
#
#  -O0                                = 5.17
#  -O0 -march=native                  = 5.20
#  -O0 -march=native -flto            = 5.21
#  -O2                                = 1.17
#  -O2 -march=native                  = 1.09
#  -O2 -flto                          = 1.07      (4.83x to -O0)
#  -O2 -march=native -flto            = 1.13
#  -O3 -march=native -flto            = 1.11
#
#  OMP_NUM_THREADS=16 : fastest 1.08, slowest 5.21
#  OMP_NUM_THREADS=8  : fastest 1.11, slowest 5.20
#  OMP_NUM_THREADS=4  : fastest 1.10, slowest 5.23 <- Tuning doesn't work?
#  OMP_NUM_THREADS=2  : fastest 1.04, slowest 5.16
#  OMP_NUM_THREADS=1  : fastest 1.09, slowest 5.23
#
# [ E5-2687W v4, g++ 6.3.0-16 Debian ]
#
#  -O0                                = 5.10
#  -O0 -march=native                  = 5.06
#  -O0 -march=native -flto            = 5.46
#  -O2                                = 1.40
#  -O2 -march=native                  = 1.46
#  -O2 -flto                          = 1.48
#  -O2 -march=native -flto            = 1.61
#  -O3 -march=native -flto            = 1.49
#
#  OMP_NUM_THREADS=12 : fastest 1.44, slowest 5.31
#  OMP_NUM_THREADS=6  : fastest 1.42, slowest 5.03
#  OMP_NUM_THREADS=4  : fastest 1.28, slowest 4.84 <- Tuning improves?
#  OMP_NUM_THREADS=2  : fastest 1.39, slowest 5.99
#
# [ E5-2687W v4, clang++ 4.0.1-1 Debian ]
#
#  -O0                                = 5.02
#  -O0 -march=native                  = 5.21
#  -O0 -march=native -flto            = 5.23
#  -O2                                = 1.13
#  -O2 -march=native                  = 1.09
#  -O2 -flto                          = 1.14
#  -O2 -march=native -flto            = 1.13
#  -O3 -march=native -flto            = 1.12
#
#  OMP_NUM_THREADS=24 : fastest 1.01, slowest 6.10
#  OMP_NUM_THREADS=16 : fastest 1.14, slowest 6.39
#  OMP_NUM_THREADS=12 : fastest 1.12, slowest 4.95
#  OMP_NUM_THREADS=8  : fastest 1.15, slowest 6.16
#  OMP_NUM_THREADS=6  : fastest 1.20, slowest 5.12
#  OMP_NUM_THREADS=4  : fastest 1.18, slowest 5.95 <- Tuning doesn't work?
#  OMP_NUM_THREADS=2  : fastest 1.25, slowest 5.78
#  OMP_NUM_THREADS=1  : fastest 1.13, slowest 6.32

clean:
	-$(RM) demo.h5 *.elf
