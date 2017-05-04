package main

import "fmt"
import "math"
import "os"

type Abser interface {
	Abs() float64
}

type Vertex struct {
	x int
	y int
}

func (v *Vertex) Abs() float64 { // go does not have classes.
	return math.Sqrt(float64(v.x*v.x + v.y*v.y))
}

// struct literals
var (
	v1 = Vertex{1, 2}
	v2 = Vertex{x: 1}
	v3 = Vertex{}
	p  = &Vertex{1, 2}
)

type MyFloat float64

func (f MyFloat) Abs() float64 {
	if f < 0 {
		return float64(-f)
	}
	return float64(f)
}

func main() {
	fmt.Println(Vertex{1, 2})

	v := Vertex{3, 4}
	fmt.Println(v.x, v.y)

	p := &v
	p.x = 1e9
	fmt.Println(v)
	p.x = 3

	fmt.Println(v.Abs())

	f := MyFloat(-math.Sqrt2)
	fmt.Println(f.Abs())

	var abser Abser
	abser = f // a MyFloat implements Abser
	fmt.Println(abser)

	fmt.Fprintf(os.Stdout, "hello stdout\n")
}
