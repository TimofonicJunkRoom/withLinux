package main

import "fmt"

type Vertex struct {
	x int
	y int
}

// struct literals
var (
	v1 = Vertex{1, 2}
	v2 = Vertex{x: 1}
	v3 = Vertex{}
	p  = &Vertex{1, 2}
)

func main() {
	fmt.Println(Vertex{1, 2})

	v := Vertex{3, 4}
	fmt.Println(v.x, v.y)

	p := &v
	p.x = 1e9
	fmt.Println(v)
}
