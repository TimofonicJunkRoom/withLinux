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
	std::vector<int> buf {123,12,11,5,7,43,7,4,7,467,1};
	vu::vector_dump(buf);

	lumin::insertion_sort(buf);
	vu::vector_dump(buf);

	lumin::insertion_sort_v2(buf);
	vu::vector_dump(buf);

	return 0;
}
