package main

import "fmt"

func main() {
	var a [2]string // its size is fixed
	a[0] = "hello"
	a[1] = "world"
	fmt.Println(a[0], a[1])
	fmt.Println(a)

	p := []int{1, 23, 45, 5, 6, 7, 9} // slice
	fmt.Println("p =", p)

	for i := 0; i < len(p); i++ {
		fmt.Printf("p[%d] == %d\n", i, p[i])
	}

	fmt.Println(p[1:3], p[5:], p[:3])

	b := make([]int, 5)    // len(b) == 5
	c := make([]int, 0, 5) // len(b) == 0, cap(b) == 5
	b = b
	c = c

	var z []int
	fmt.Println(z, len(z), cap(z))
	if z == nil {
		fmt.Println("nil!")
	}
	z = append(z, 1)
	z = append(z, 2, 3, 4, 5, 6)
	fmt.Println(z)
}
