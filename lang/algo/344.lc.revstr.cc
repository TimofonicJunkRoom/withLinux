class Solution {
public:
    string reverseString(string s) {
            string ret;
    ret.clear();
    for (unsigned int i = s.length(); i > 0; i--) {
      ret.append(1, s.at(i-1));
    }
    return ret;

    }
};
