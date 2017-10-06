#include <iostream>
#include <vector>
using namespace std;

// Bubble sort. O(n^2), Stable.
void bsort(vector<int>& v) {
	for (int i = 0; i < (int)v.size(); i++) {
		// find the min value in v_i, i \in [i, v.size-1]
		// i.e. find the i-th min value, then put it at i
		int idxmin = i;
		for (int j = i; j < (int)v.size(); j++) {
			idxmin = (v[j] < v[idxmin]) ? j : idxmin;
		}
		// swap
		int tmp = v[i];
		v[i] = v[idxmin];
		v[idxmin] = tmp;
	}
}

int
main(void)
{
	vector<int> v {34,65,12,43,67,5,78,10,3,3,70};
	cout << "orig seq" << endl;
	for (auto i : v) cout << " " << i;
	cout << endl;

	bsort(v);
	cout << "sort seq" << endl;
	for (auto i : v) cout << " " << i;
	cout << endl;
	return 0;
}
