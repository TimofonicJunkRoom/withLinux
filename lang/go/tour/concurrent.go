package main

import "fmt"
import "time"

func say(s string) {
	for i := 0; i < 5; i++ {
		time.Sleep(100 * time.Millisecond)
		fmt.Println(s)
	}
}

func sum(a []int, c chan int) {
	sum := 0
	for _, v := range a {
		sum += v
	}
	c <- sum
}

func fibonacci(n int, c chan int) {
	x, y := 0, 1
	for i := 0; i < n; i++ {
		c <- x
		x, y = y, x+y
	}
	close(c)
}

func fibonacci2(c, quit chan int) {
	x, y := 0, 1
	for {
		select {
		case c <- x:
			x, y = y, x+y
		case <-quit:
			fmt.Println("quit")
			return
		}
	}
}

func main() {
	go say("world")
	say("hello")

	a := []int{1, 32, 43, 4, 243, -7}
	c := make(chan int) // make(chan int, 100) for buffered channel
	go sum(a[:len(a)/2], c)
	go sum(a[len(a)/2:], c)
	x, y := <-c, <-c // receive from c
	fmt.Println(x, y, x+y)

	ch := make(chan int, 20)
	go fibonacci(cap(ch), ch)
	for i := range ch {
		fmt.Println(i)
	}

	cc, qq := make(chan int), make(chan int)
	go func() {
		for i := 0; i < 10; i++ {
			fmt.Println(<-cc)
		}
		qq <- 0
	}()
	fibonacci2(cc, qq)
}
