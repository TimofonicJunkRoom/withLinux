;!/usr/bin/scm
(print "http://www.shido.info/lisp/idx_scm_e.html")
; this is a comment

; --- scheme as a calculator ---
(print (+ 1 2))
(print (+ 1 2 3 4))
(print (- 10 3))
(print (- 10 3 2))
(print (* 2 3 4))
(print (/ 9 6))

(print (* (+ 1 39) (- 53 45)))
(print (+ (/ 1020 39) (* 45 2)))
(print (+ 39 48 72 23 91))
(print (/ (+ 39 48 72 23 91) 5.0))

(print (quotient 7 3)) ; 2
(print (modulo 7 3)) ; 1
(print (sqrt 8))

(print (* 4 (atan 1)))
(print (exp 1))
(print (log 2.71828))

; --- lists ---
(print (cons 1 2)); 1 for car, 2 for cdr
(print '()); '() is a list

(print (cons "hi" "everybody"))
(print (cons 0 '()))
(print (cons 1 (cons 10 100)))
(print (cons 1 (cons 10 (cons 100 '())))); this is a list
(print '(+ 1 2))
(print (car '(1 2 3 4)))
(print (cdr '(1 2 3 4)))

(print (list))
(print (list 1 2 3 4))
(print (list '(1 2) '(3 4)))

; --- functions ---
; (cd "/tmp") ; chdir
; (load "abc.scm") ; load another scheme file
(define vhello "hello world")
(define fhello (lambda () (print "hello world")))
(print vhello)
(fhello)

(define hello (lambda (name) (string-append "hello " name)))
; (define (hello name) (string-append "hello " name))
(define sum3 (lambda (a b c) (+ a b c)))
; (define (sum3 a b c) (+ a b c))
(print (hello "lisp"))
(define (add1 x) (+ x 1))
(define (sub1 x) (- x 1))

; --- branch ---
; (if predicate then_value else_value)
(print (null? '()))
(print (null? '(1 2 3 4)))

(and #f 0); '()
(and 1 2 3); 3
(and 1 2 3 #f); '()

(+ (cond
  (#t 1)
  (#f 0)))

(eq? 1 1)
(eqv? 1 1.0)
(equal? (list 1 2 3) (list 1 2 3))

; functions that checks data type:
;  pair? list? null? symbol? char? string? number? complex? real? rational?
;  interger? exact? inexact?
; functions that compares number:
;  = > < >= <= odd? even? positive? negative? zero?

; --- local variable ---
;(let binds body)
(print (let ((i 1) (j 1)) (+ i j)))

; --- repetition ---
(define (factorial n)
  (if (= n 1) 1 (* n (factorial (- n 1)))))
(print (factorial 3))

; --- high order functions ---
; mapping: (map ...)
; filtering
; folding: (reduce ...)
; ...

; ... TODO ...

(quit)
