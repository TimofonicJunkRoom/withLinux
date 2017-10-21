#include <iostream>
#include <queue>

int
main(void)
{
	std::queue<int> q;
	for (int i = 1; i <= 7; i++) q.push(i);

	while (!q.empty()) {
		std::cout << q.front() << " ";
		q.pop();
		q.push(q.front());
		q.pop();
	}
	return 0;
}
