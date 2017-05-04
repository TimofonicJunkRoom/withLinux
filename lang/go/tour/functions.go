package main

import (
	"fmt"
	"math"
)

func add(x int, y int) int {
	return x + y
}

func swap(x, y string) (string, string) {
	return y, x
}

func adder() func(int) int {
	sum := 0
	return func(x int) int { // returns a closure
		sum += x
		return sum
	}
}

func main() {
	fmt.Println(add(42, 13))

	a, b := swap("hello", "world")
	fmt.Println(a, b)

	hypot := func(x, y float64) float64 { // functions are values too
		return math.Sqrt(x*x + y*y)
	}
	fmt.Println(hypot(3, 4))

	pos, neg := adder(), adder() // each closure has its own sum
	for i := 1; i < 10; i++ {
		fmt.Println(
			pos(i),
			neg(-2*i),
		)
	}
}
