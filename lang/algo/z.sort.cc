/* 
 * @brief A collection of sorting algorithms.
 * Input  : a vector of n numbers <a1, a2, ..., an>
 * Output : a permutation of original vector
 */
#include <iostream>
#include <vector>
#include "helper.hpp"
using namespace std;

namespace sort {

// List os Sorting functions here.
void selSort(vector<int>&);     // kind: Selection
void naiveSort(vector<int>&);   // kind: Selection
void qSort(vector<int>&);       // kind: Swapping
void insSort(vector<int>&);     // kind: Insertion

// Selective sort. O(n^2), Unstable. Ascending.
// suitable for small arrays.
void
selSort(vector<int>& v) {
    for (int i = 0; i < (int)v.size(); i++) {
        // find the mininum v_i for i in range [i, v.size)
        int idxmin = i;
        for (int j = i; j < (int)v.size(); j++) {
            idxmin = (v[j] < v[idxmin]) ? j : idxmin;
        }
        swap(v[i], v[idxmin]);
    }
}

// naive sort, degradation of Selective sort.
void
naiveSort(vector<int>& v) {
    for (int i = 0; i < v.size(); i++)
        for (int j = i; j < v.size(); j++)
            if (v[j] < v[i]) swap(v[j], v[i]);
}

// Quick sort. O(n logn) in the best case. O(n^2) in the worst case.
void
_qSort(vector<int>& v, int curl, int curr) {
	if (curl < curr) {
		int i = curl, j = curr, pivot = v[i];
		while (i < j) {
			while (i < j && v[j] > pivot) j--;
			if (i < j) v[i++] = v[j];
			while (i < j && v[i] < pivot) i++;
			if (i < j) v[j--] = v[i];
		}
		v[i] = pivot;
		_qSort(v, curl, i-1);
		_qSort(v, i+1, curr);
	}
}
void qSort(vector<int>& v) { return _qSort(v, 0, v.size()-1); }

// Insertion Sort, inplace
void
insSort (std::vector<int>& v)
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

} // namespace sort

#define TEST(name, sortfun) do { \
	std::cout << "=> Test " << name << std::endl; \
	std::vector<int> v1 {34,65,12,43,67,5,78,10,3,3,70}; \
	std::vector<int> v2 {123,12,11,5,7,43,7,4,7,467,1}; \
	std::vector<int> v3 {1,0,0,1,0,1,1,1,1,0,0,1,0}; \
	std::cout << "  :: Orig " << v1 << " -> Sorted "; \
	sortfun(v1); std::cout << v1 << std::endl; \
	std::cout << "  :: Orig " << v2 << " -> Sorted "; \
	sortfun(v2); std::cout << v2 << std::endl; \
	std::cout << "  :: Orig " << v3 << " -> Sorted "; \
	sortfun(v3); std::cout << v3 << std::endl; \
} while(0)

int
main(void)
{
	TEST("Selective Sort", sort::selSort);
	TEST("Naive Sort", sort::naiveSort);
	TEST("Quick Sort", sort::qSort);
	TEST("Insertion Sort", sort::insSort);
    return 0;
}
