package main

import "fmt"

type Person struct {
	Name string
	Age  int
}

type IPAddr [4]byte

func (p Person) String() string {
	return fmt.Sprintf("[%v] %v ", p.Name, p.Age)
}

func (ip IPAddr) String() string {
	return fmt.Sprintf("%v.%v.%v.%v", ip[0], ip[1], ip[2], ip[3])
}

func main() {
	a := Person{"Arthur Dent", 42}
	fmt.Println(a)

	ip := IPAddr{8, 8, 8, 8}
	fmt.Println(ip)
}
