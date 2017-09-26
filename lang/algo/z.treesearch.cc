#include <iostream>
#include <vector>

void treeSearch(std::vector<int>&, int);
int pcount = 0; // print count

int
main(void)
{
	std::vector<int> v(6, 0);
	treeSearch(v, 0);
	std::cout << "dump pcount " << pcount << std::endl;
	return 0;
}

void treeSearch(std::vector<int>& buf, int cur) {
	if (cur == buf.size()) {
		for (auto i: buf) std::cout << ' ' << i;
		std::cout << std::endl;
		pcount++;
	} else {
		for (int i = 0; i < 2; i++) {
			buf[cur] = i;
			treeSearch(buf, cur+1);
		}
	}
}
