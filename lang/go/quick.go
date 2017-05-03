// reference: https://learnxinyminutes.com/docs/go/
// single line comment
/* multi-line
   comments */

// main is a special package, declaring an executable rather than a lib
package main

import (
	"fmt"
	"io/ioutil"
	m "math"
	"net/http"
	"os"
	"strconv"
)

func main() {
	fmt.Println("Hello world!")
	beyondHello()
}

func beyondHello() {
	var x int
	x = 3
	y := 4 // short declaration
	sum, prod := learnMultiple(x, y)
	fmt.Println("sum:", sum, "prod:", prod)
	learnTypes()
}

// x and sum receive type "int"
func learnMultiple(x, y int) (sum, prod int) {
	return x+y, x*y
}

func learnTypes() {
	// Short
	str := "Learn GO!"
	s2 := `a "raw" string leteral
	can include line breaks.`
	g := 'Σ' // non-ASCII, rune type (int32)
	f := 3.14159 // float64
	c := 3+4i // complex128
	// Var syntax
	var u uint = 7
	var pi float32 = 22. / 7
	n := byte('\n') // byte is alias of uint8
	var a4 [4]int // an array of 4 ints, all 0
	a3 := [...]int{3, 1, 5} // 3-element array
	s3 := []int{4,5,9} // slices have dynamic size
	s4 := make([]int, 4) // allocate slice of 4 ints, all 0
	var d2 [][]float64 // nothing allocated
	bs := []byte("a slice") // type convertion
	s := []int{1,2,3} // slice of length 3
	s = append(s, 4,5,6) // slice now has length of 6
	fmt.Println(s)
	s = append(s, []int{7,8,9}...) // unpack ...
	fmt.Println(s)

	p, q := learnMemory()
	fmt.Println(*p, *q) // * follows a pointer
	m := map[string]int{"three":3, "four":4}
	m["one"] = 1

	// unused variables are error
	_, _, _, _, _, _, _, _, _, _ = str, s2, g, f, u, pi, n, a3, s4, bs

	file, _ := os.Create("output.txt")
	fmt.Fprint(file, "this is how you write to a file")
	file.Close()
	fmt.Println(s, c, a4, s3, d2, m)

	learnFlowControl()
}

func learnNamedReturns(x, y int) (z int) {
	z = x*y
	return // z is returned
}

func learnMemory() (p, q *int) {
	p = new(int) // init 0
	s := make([]int, 20)
	s[3] = 7
	r := -2
	return &s[3], &r // & takes the address of an object
}

func expensiveComputation() float64 {
	return m.Exp(10)
}

learnFlowControl() {
	if true {
		fmt.Println("told ya")
	} else {
		// nothing
	}

	x := 42.0
	switch x {
	case 0:
	case 1:
	case 42:
		// cases don't fall through
	case:43:
		// unreached
	default:
		// optional
	}

	for x := 0; x < 3; x++ {
		fmt.Println("iteration", x)
	}

	for { // infinite loop
		break // just kidding
		continue // unreached
	}
	// for is the only loop statement in Go

	for key, value := range map[string]int{"one":1, "two":2} {
		fmt.Printf("key=%s, value=%d\n", key, value)
	}

	for _, name := range []string{"bob", "bill"} {
		fmt.Printf("Hello, %s\n", name)
	}

	if y := expensiveComputation(); y > x {
		x = y
	}

	xBig := func() bool { // closure
		return x > 10000 // references x declared above
	}
	x = 999999
	fmt.Println("xBig:", xBig())
	x = 1.3e3
	fmt.Println("xBig:", xBig())

	fmt.Println("add + double two numbers: ",
		func(a,b int) int {
			return (a+b)*2
		}(10,2))

	goto love
love:
	learnFunctionFactory()
	learnDefer()
	learnInterfaces()
}

func learnFunctionFactory() {
	fmt.Println(sentenceFactory("summer")("A beautiful", "day"))
	d := sentenceFactory("summer")
	fmt.Println(d("A beautiful", "day"))
	fmt.println(d("A lazy", "afternoon"))
}

func sentenceFactory(mystring string) func(before, after string) string {
	return func(before, after string) string {
		return fmt.Sprintf("%s %s %s", before, mystring, after)
	}
}

func learnDefer() (ok bool) {
	defer fmt.Println("deferred statements execute in LIFO order")
	defer fmt.Println("\nThis line is being printed first because")
	return true
}

type Stringer interface {
	String() string
}

type pair struct {
	x, y int
}

func (p pair) String() string { // p is called the reciever
	return fmt.Sprintf("%d %d", p.x, p.y)
}

func learnInterfaces() {
	p := pair{3,4}
	fmt.Println(p.String())
	var i Stringer
	i = p
	fmt.Println(i.String())

	fmt.Println(p)
	fmt.Println(i)
	learnVariadicParams("great", "learning", "here!")
}

func learnVariadicParams(myStrings ...interface{}) {
	for _, param := range myStrings {
		fmt.Println("param:", param)
	}
	fmt.Println("params:", fmt.Sprintln(myStrings...))

	learnErrorHandling()
}

func learnErrorHandling() {
	m := map[int]string{3: "three", 4: "four"}

	if x, ok := m[1]; !ok { // "; ok" idiom
		fmt.Println("no one there")
	} else {
		fmt.Println(x)
	}

	if _, err := strconv.Atoi("non-int"); err != nil {
		fmt.Println(err)
	}
	learnConcurrency()
}

func inc(i int, c chan int) {
	c <- i + 1 // <- is the "send" operator when a channel appears on the left
}

func learnConcurrency() {
	c := make(chan int)
	go inc(0, c) // "go" starts a new goroutine
	go inc(10, c)
	go inc(-805, c)
	fmt.Println(<-c, <-c, <-c) // <- is the receive operator

	cs := make(chan string)
	ccs := make(chan chan string)
	go func() { c <- 84 }()
	go func() { cs <- "wordy" }()
	select {
	case i := <-c:
		fmt.Printf("it's a %T", i)
	case <- cs:
		fmt.Println("it's a string")
	case <- ccs:
		fmt.Println("didn't happen.")
	}

	learnWebProgramming()
}

func learnWebProgramming() {
	go func() {
		err := http.ListenAndServe(":8080", pair{})
		fmg.Println(err)
	}()

	requestServer()
}

func (p pair) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	w.Write([]byte("You learned Go in Y minutes"))
}

func requestServer() {
	resp, err := http.Get("http://localhost:8080")
	fmt.Println(err)
	defer resp.Body.Close()
	body, err := ioutil.ReadAll(resp.Body)
	fmt.Printf("\nWebserver said: %s", string(body))
}
