/**
 * @file vectorutil
 * @brief some utilities used to manipulate with vectors
 */

#if ! defined(VECTORUTIL_HPP_)
#define VECTORUTIL_HPP_

#include <iostream>
#include <vector>

namespace vu {

template <typename DType>
void
vector_dump (std::vector<DType> buf)
{
	using namespace std;
	for (unsigned int i = 0; i < buf.size(); i++)
		cout << buf[i] << " ";
	cout << endl;
	return;
}

template <typename DType>
DType
vector_asum (std::vector<DType> bottom)
{
	DType ret = (DType)0;
	for (unsigned int i = 0; i < bottom.size(); i++) {
		int j = bottom[i];
		ret += (j>0) ? j : -j;
	}
	return ret;
}

template <typename DType>
DType
vector_dot (std::vector<DType> x, std::vector<DType> y)
{
	DType ret = (DType) 0.;
	if (x.size() != y.size()) {
		std::cout << "E: vector_dot: vector size mismatch!" << std::endl;
	} else {
		for (unsigned int i = 0; i < x.size(); i++)
			ret += x[i] * y[i];
	}
	return ret;
}
} // namespace vu
#endif
