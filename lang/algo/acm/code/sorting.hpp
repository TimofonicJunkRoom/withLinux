/* sorting.hpp
 * 
 * Input  : a vector of n numbers <a1, a2, ..., an>
 * Output : a permutation of original vector
 */

/* SORTING
 *  * insertion sort
 *  * merge sort
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

// in-place insertion sort
// reference: Intro to Algo, pp. 18
void
insertion_sort_v2 (std::vector<int> & v)
{
	if (v.size() < 2) return;
	for (unsigned int i = 2; i < v.size(); i++) {
		int pivot = v.at(i);
		// do insertion
		unsigned int j = i - 1;
		while ((j>0) && v.at(j) > pivot) {
			v.at(j+1) = v.at(j);
			j--;
		}
		v.at(j+1) = pivot;
	}
	return;
}

// void merge_sort (std::vector<int> &v);

} // namespace lumin
#endif // defined SORTING_HPP
