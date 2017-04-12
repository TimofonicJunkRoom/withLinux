
double My_variable = 3.0;

int
factorial(int n) {
	if (n<=1)
		return 1;
	else
		return n*factorial(n-1);
}

int
my_mod(int n, int m) {
	return (n % m);
}
