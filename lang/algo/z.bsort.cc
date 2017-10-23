#include <iostream>
#include <vector>
#include "helper.hpp"
using namespace std;

// Bubble sort (Selective sort). O(n^2), Stable. Ascending i.e. Min -- Max
void
bsort(vector<int>& v) {
    for (int i = 0; i < (int)v.size(); i++) {
        // find the min value in v_i, i \in [i, v.size-1]
        // i.e. find the i-th min value, then put it at i
        int idxmin = i;
        for (int j = i; j < (int)v.size(); j++) {
            idxmin = (v[j] < v[idxmin]) ? j : idxmin;
        }
        swap(v[i], v[idxmin]);
    }
}

// Bubble sort
void
bsort_v2(vector<int>& v) {
    for (int i = v.size()-1; i >= 0; i--)
        for (int j = 0; j <= i ; j++)
            if (v[j] > v[i]) swap(v[j], v[i]);
}

// bubble sort
void bsort_v3(vector<int>& v) {
    for (int i = 0; i < v.size(); i++)
        for (int j = i; j < v.size(); j++)
            if (v[j] < v[i]) swap(v[j], v[i]);
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

    vector<int> v2 {34,65,12,43,67,5,78,10,3,3,70};
    bsort_v2(v2);
    cout << v2;

    vector<int> v3 {34,65,12,43,67,5,78,10,3,3,70};
    bsort_v3(v3);
    cout << v3;

    return 0;
}
