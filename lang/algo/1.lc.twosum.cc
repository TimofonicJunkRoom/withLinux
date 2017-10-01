#include <vector>
#include <iostream>
#include <map>

using namespace std;
#include "helper.hpp"

class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        /*
        // prepare map: value -> location
        map<int, int> m;
        for (int i = 0; i < nums.size(); i++) {
            m[nums[i]] = i;
        } // O(n)

        // searching
        for (int i = 0; i < nums.size(); i++) {
            auto cursor = m.find(target - nums[i]);
            if (cursor == m.end() || cursor->second == i) { // ?
                continue;
            } else {
                return vector<int> {i, m.find(target-nums[i])->second};
            }
        }
        */
        // assume that input is valid
        map<int, int> m;
        map<int, int>::iterator cur;
        for (int i = 0; i < nums.size(); i++) {
            if ((cur = m.find(target-nums[i])) != m.end())
                return vector<int> {i, cur->second};
			m.insert(pair<int,int>(nums[i], i));
        }
        return vector<int>{-1, -1};
    }
};

int
main(void)
{
	auto s = Solution();
	vector<int> v {3, 2, 4};
	cout << s.twoSum(v, 6) << endl;
	return 0;
}
