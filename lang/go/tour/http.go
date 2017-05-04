package main

import "log"
import "fmt"
import "net/http"

type Hello struct{}
type String string

func (h Hello) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	fmt.Fprint(w, "Hello!")
}

func (s String) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	fmt.Fprint(w, s)
}

func main() {
	var h Hello
	_ = h
	http.Handle("/hi", String("Hi!"))
	//err := http.ListenAndServe("localhost:4000", h) // h will be router
	err := http.ListenAndServe("localhost:4000", nil)
	if err != nil {
		log.Fatal(err)
	}
}
