/**
 * @file z.seqsearch.cc
 * @brief implement sequencial searching
 */
#include <iostream>
#include <vector>
#include "helper.hpp"

/**
 * @brief sequential search
 */
template <typename DType>
int
sequentialSearch(const std::vector<DType>& v, DType target)
{
	for (int i = 0; i < v.size(); i++)
		if (v[i] == target) return i;
	return -1;
}

/**
 * @brief test sequential search
 */
int
main (int argc, char ** argv)
{
	//int i;
	//while (std::cin >> i) buf.push_back(i);
	std::vector<int> buf {1,2,3,4,5,6,7,8,9};
	std::cout << sequentialSearch(buf, 5) << std::endl;
	std::cout << sequentialSearch(buf, 10) << std::endl;
	return 0;
}
