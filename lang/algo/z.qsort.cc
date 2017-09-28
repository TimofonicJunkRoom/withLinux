#include <iostream>
#include <vector>
using namespace std;

// Quick sort. O(n logn) in the best case. O(n^2) in the worst case.
void qsort(vector<int>& v, int curl, int curr) {
	if (curl < curr) {
		int i = curl, j = curr, pivot = v[i];
		while (i < j) {
			while (i < j && v[j] > pivot) j--;
			if (i < j) v[i++] = v[j];
			while (i < j && v[i] < pivot) i++;
			if (i < j) v[j--] = v[i];
		}
		v[i] = pivot;
		qsort(v, curl, i-1);
		qsort(v, i+1, curr);
	}
}

int
main(void)
{
	vector<int> v {34,65,12,43,67,5,78,10,3,3,70};
	cout << "orig seq" << endl;
	for (auto i : v) cout << " " << i;
	cout << endl;

	qsort(v, 0, v.size()-1);
	cout << "orig seq" << endl;
	for (auto i : v) cout << " " << i;
	cout << endl;
	return 0;
}
