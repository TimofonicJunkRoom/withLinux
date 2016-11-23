/* Time limite succeed
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        for (int i = 0; i < nums.size(); i++) {
            for (int j = 0; j < nums.size(); j++) {
                if (i == j) {
                    continue;
                } else {
                    if (nums.at(i)+nums.at(j) == target) {
                        return vector<int> {i, j};
                    }
                }
            }
        }
        return vector<int> {-1, -1};
    }
};
*/


class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        map<int, int> m;
        // prepare map
        for (int i = 0; i < nums.size(); i++) {
            m[nums.at(i)] = i;
        }
        // search
        map<int, int>::iterator cursor;
        for (int i = 0; i < nums.size(); i++) {
            int expected = target - nums.at(i);
            cursor = m.find(expected);
            if (cursor == m.end()) continue;
            else if (cursor->second == i) continue;
            else {
                return vector<int> {i, cursor->second};
            }
        }
        return vector<int> {-1, -1};
    }
};
