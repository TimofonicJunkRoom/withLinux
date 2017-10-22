/**
 * @file helper.hpp
 * @brief misc helper functions including printing, etc.
 */
#if ! defined(HELPER_HPP_)
#define HELPER_HPP_

#include <iostream>
#include <vector>
#include <cassert>

//https://stackoverflow.com/questions/10750057/how-o-print-out-the-contents-of-a-vector

/* 1D vector dump */
template <typename T>
std::ostream&
operator<< (std::ostream& out, const std::vector<T>& v) {
	out << "[";
	for (auto i : v) out << i << ", ";
	out << "\b\b]" << std::endl;
	return out;
}

/* 2D vector (matrix) dump */
template <typename T>
std::ostream&
operator<< (std::ostream& out,
		const std::vector<std::vector<T>>& m) {
	out << "[" << std::endl;
	for (auto v : m) {
		out << "  " << v;
	}
	out << "]" << std::endl;
	return out;
}

/* old dumping function */
template <typename DType>
void
xvdump (std::vector<DType> buf)
{
	using namespace std;
	for (unsigned int i = 0; i < buf.size(); i++)
		cout << buf[i] << " ";
	cout << endl;
	return;
}

/* x-typed vector absolute sum, b = \sum_i abs(a_i) */
template <typename DType>
DType
xvasum (std::vector<DType> bottom)
{
	DType ret = (DType)0;
	for (unsigned int i = 0; i < bottom.size(); i++) {
		int j = bottom[i];
		ret += (j>0) ? j : -j;
	}
	return ret;
}

/* x-typed vector dot product, c = \sum_i a_i * b_i */
template <typename DType>
DType
xvdot (std::vector<DType> x, std::vector<DType> y)
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
#endif // HELPER_HPP_
