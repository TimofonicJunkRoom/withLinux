/* 
 * @brief A collection of sorting algorithms.
 * Input : a vector of n numbers <a1, a2, ..., an>
 * Output: None, the given vector is sorted. Ascending.
 * @ref http://www.cnblogs.com/kkun/archive/2011/11/23/2260312.html
 */
#include <iostream>
#include <vector>
#include <algorithm>
#include "helper.hpp"
using namespace std;

namespace sort {

/* ----------------------------------------------- List of Sorting functions */

template <typename DType>
void selSort(vector<DType>&);     // kind: Selection
template <typename DType>
void naiveSort(vector<DType>&);   // kind: Selection
//void heapSort(vector<int>&);    // kind: Selection

template <typename DType>
void bSort(vector<DType>&);       // kind: Swapping
template <typename DType>
void qSort(vector<DType>&);       // kind: Swapping

template <typename DType>
void insSort(vector<DType>&);     // kind: Insertion
//void shellSort(vector<int>&);   // kind: Insertion

//void mSort(vector<int>&);       // kind: Merge
//void radixSort(vector<int>&);   // kind: Radix
template <typename DType>
void naiveBucketSort(vector<DType>&);  // kind: Bucket

/* ------------------------------------------- END List of Sorting functions */

// Bucket Sort. Stable. Very Fast. Memory Consuming.
template <typename DType> // DType \in {Int, Long}
void
naiveBucketSort(vector<DType>& v) {
	if (v.empty()) return;
	DType vmin = v[0], vmax = v[0];
	for (auto i : v) { vmin = min(vmin, i); vmax = max(vmax, i); }
	vector<int> bucket (vmax-vmin+1, 0);
	for (auto i : v) bucket[i-vmin]++;
	int cursor = 0;
	for (int i = 0; i < bucket.size(); i++)
		while (bucket[i]-- > 0) v[cursor++] = i + vmin;
}

// Bubble Sort. Stable.
template <typename DType>
void
bSort(vector<DType>& v) {
	for (int i = 0; i < v.size(); i++) {
		bool dirty = false;
		for (int j = v.size()-1; j > i; j--) {
			if (v[j] < v[j-1]) {
				dirty = true;
				swap(v[j], v[j-1]);
			}
		}
		if (!dirty) break;
	}
}

// Selective Sort. O(n^2), Unstable. Ascending.
// suitable for small arrays.
template <typename DType>
void
selSort(vector<DType>& v) {
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
template <typename DType>
void
naiveSort(vector<DType>& v) {
    for (int i = 0; i < v.size(); i++)
        for (int j = i; j < v.size(); j++)
            if (v[j] < v[i]) swap(v[j], v[i]);
}

// Quick sort. O(n logn) in the best case. O(n^2) in the worst case.
template <typename DType>
void
_qSort(vector<DType>& v, int curl, int curr) {
	if (curl < curr) {
		int i = curl, j = curr;
		DType pivot = v[i];
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
template <typename DType>
void qSort(vector<DType>& v) { return _qSort(v, 0, v.size()-1); }

// Insertion Sort, inplace
template <typename DType>
void
insSort (std::vector<DType>& v)
{
	if (v.empty()) return;
	for (int i = 0; i < v.size(); i++) { // i-1: sorted length 
		// find insert position, move v[i] to that position
		DType pivot = v[i];
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
	std::vector<int> v4 {100, 10}; \
	std::cout << "  :: Orig " << v1 << " -> Sorted "; \
	sortfun(v1); std::cout << v1 << std::endl; \
	std::cout << "  :: Orig " << v2 << " -> Sorted "; \
	sortfun(v2); std::cout << v2 << std::endl; \
	std::cout << "  :: Orig " << v3 << " -> Sorted "; \
	sortfun(v3); std::cout << v3 << std::endl; \
	std::cout << "  :: Orig " << v4 << " -> Sorted "; \
	sortfun(v4); std::cout << v4 << std::endl; \
} while(0)

int
main(void)
{
	TEST("Selective Sort", sort::selSort);
	TEST("Naive Sort", sort::naiveSort);
	TEST("Quick Sort", sort::qSort);
	TEST("Insertion Sort", sort::insSort);
	TEST("Bubble Sort", sort::bSort);
	TEST("Naive Bucket Sort", sort::naiveBucketSort);
    return 0;
}
