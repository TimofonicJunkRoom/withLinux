package main

import "log"
import "runtime"

func main() {
	log.SetFlags(log.Lshortfile | log.LstdFlags)
	log.Println("hhh")

	funcName, file, line, ok := runtime.Caller(0)
	if ok {
		log.Println("funcName=", runtime.FuncForPC(funcName).Name())
		log.Println("file=", file)
		log.Println("line=", line)
	}
}
