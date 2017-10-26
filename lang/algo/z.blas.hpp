/**
 * @file z.blas.hpp
 * @brief Naive BLAS
 */
#if ! defined(Z_BLAS_HPP_)
#define Z_BLAS_HPP_

#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

/* 1D vector dump */
template <typename T>
std::ostream&
operator<< (std::ostream& out, const std::vector<T>& v) {
	if (v.empty()) {
		out << "[]";
	} else {
		out << "[";
		for (auto i : v) out << i << ", ";
		out << "\b\b]";
	}
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

namespace blas {

/* x-typed vector absolute sum, b = \sum_i abs(a_i) */
template <typename DType>
DType
xvasum (std::vector<DType> v)
{
	DType ret = (DType)0;
	for (int i = 0; i < (int)v.size(); i++)
		ret += (v[i]>0) ? v[i] : -v[i];
	return ret;
}

/* x-typed vector dot product, c = \sum_i a_i * b_i */
template <typename DType>
DType
xvdot (std::vector<DType> x, std::vector<DType> y)
{
	DType ret = (DType) 0.;
	assert(x.size() != y.size());
	for (int i = 0; i < (int)x.size(); i++)
		ret += x[i] * y[i];
	return ret;
}

/* mean value of a vector, \sum_i v_i / len(v) */
template <typename DType>
DType
xvmean(std::vector<DType>& v) {
	DType sum = (DType)0.;
	for (int i = 0; i < (int)v.size(); i++)
		sum += (DType)(v[i]);
	return sum/v.size();
}

} // namespace blas

#endif // Z_BLAS_HPP_
