main:
	flex ex.l
	gcc lex.yy.c -Wall
	./a.out < test
	./a.out < matrix
	./a.out < mat2
clean:
	-rm a.out lex.yy.c
