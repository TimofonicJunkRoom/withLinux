#include <iostream>
#include <vector>

//https://stackoverflow.com/questions/10750057/how-o-print-out-the-contents-of-a-vector
template <typename T>
std::ostream&
operator<< (std::ostream& out, const std::vector<T>& v) {
	out << "[";
	for (auto i : v) out << i << ", ";
	out << "\b\b]";
	return out;
}
