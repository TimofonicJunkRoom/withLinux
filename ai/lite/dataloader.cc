/* Dataloader.cc */

#include <string>
#include <cassert>
#include <cstring>
#include <iostream>
#include <cstdio>

#include "H5Cpp.h"

using namespace std;
using namespace H5;

// read the whole dataset into memory
template <typename Dtype>
void
lite_hdf5_read(
		H5std_string name_h5file,
		H5std_string name_dataset,
		Dtype* dest)
{
	H5File h5file (name_h5file, H5F_ACC_RDONLY);
	DataSet dataset = h5file.openDataSet(name_dataset);
	H5T_class_t type_class = dataset.getTypeClass();
	DataSpace dataspace = dataset.getSpace();

	hsize_t offset[2] = {0, 0};
	hsize_t count[2] = {8, 17};
	dataspace.selectHyperslab(H5S_SELECT_SET, count, offset);

	hsize_t dimsm[2] = {8, 17};
	DataSpace memspace(2, dimsm);

	hsize_t offset_out[2] = {0, 0};
	hsize_t count_out[2] = {8, 17};
	memspace.selectHyperslab(H5S_SELECT_SET, count_out, offset_out);

	dataset.read(dest, PredType::NATIVE_DOUBLE, memspace, dataspace);
}

int
main()
{
	double data_out[8][17];
	memset(data_out, 0x0, 8*17*sizeof(double));
	lite_hdf5_read("demo.h5", "data", data_out);
	for (int i = 0; i < 8; i++) {
		for (int j = 0; j < 17; j++) {
			printf(" %7.4f", data_out[i][j]);
		}
		cout << endl;
	}
	return 0;
}
