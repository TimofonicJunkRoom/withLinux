/* 
 * Input  : a vector of n numbers <a1, a2, ..., an>
 * Output : a permutation of original vector
 */

#include <iostream>
#include <vector>
#include "helper.hpp"
using namespace std;

// in-place insertion sort
void
insertion_sort (std::vector<int>& v)
{
	if (v.empty()) return;
	for (int i = 0; i < v.size(); i++) { // i-1: sorted length 
		// find insert position, move v[i] to that position
		int pivot = v[i];
		int j = i;
		while(j > 0 && pivot < v[j-1]) {
			v[j] = v[j-1];
			j--;
		}
		v[j] = pivot;
	}
	return;
}

int
main(void)
{
	std::vector<int> buf {123,12,11,5,7,43,7,4,7,467,1};
	cout << buf << endl;
	insertion_sort(buf);
	cout << buf << endl;

	return 0;
}
