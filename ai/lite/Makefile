demo.h5:
	python3 gen-demoh5.py
dataloader: demo.h5
	g++ dataloader.cc -I/usr/include/hdf5/serial -lhdf5_cpp -lhdf5_hl_cpp -lhdf5_serial -o dataloader_test -DLITE_TEST
	./dataloader_test
tensor: demo.h5
	g++ tensor.cc -I/usr/include/hdf5/serial -lhdf5_cpp -lhdf5_hl_cpp -lhdf5_serial -o tensor_test -DLITE_TEST_TENSOR
	./tensor_test
clean:
	-$(RM) dataloader_test demo.h5
