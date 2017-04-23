/* sorting.hpp
 * 
 * Input  : a vector of n numbers <a1, a2, ..., an>
 * Output : a permutation of original vector
 */

#if ! defined(SORTING_HPP_)
#define SORTING_HPP_

#include <iostream>
#include <vector>

namespace lumin {

// in-place insertion sort
void
insertion_sort (std::vector<int> & v)
{
	using namespace std;
	for (unsigned int i = 1; i < v.size(); i++) { // handle this number
		// select insert location
		unsigned int j = 0;
		while ((j < i) && (v.at(i) > v.at(j))) j++;
		// now j is the target location, move i to j
		int tmp = v.at(i);
		v.erase(v.begin()+i);
		v.insert(v.begin()+j, tmp);
	}
	return;
}

} // namespace lumin
#endif // defined SORTING_HPP
