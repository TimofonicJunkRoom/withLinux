package main

import (
	"fmt"
)

func naiveSort(v []int) {
	for i := 0; i < len(v); i++ {
		for j := i; j < len(v); j++ {
			if v[j] < v[i] {
				v[i], v[j] = v[j], v[i]
			}
		}
	}
}

func main() {
	var v []int = []int{1, 4, 3, 2, 6, 7, 8, 5, 3, 2, 5}
	fmt.Println(v)
	naiveSort(v)
	fmt.Println(v)
}
