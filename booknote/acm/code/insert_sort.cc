/**
 * @file insert.cc
 * @brief implement insert sorting
 */

#include <iostream>
#include <vector>
#include <cassert>

#include "vectorutil.hpp"
#include "sorting.hpp"

int debug = 1;

int
main (void)
{
	std::vector<int> buf;
	{ // input and dump
		int i;
		while (std::cin >> i) buf.push_back(i);
		vu::vector_dump(buf);
	}
	lumin::insertion_sort(buf);
	vu::vector_dump(buf);

	return 0;
}
