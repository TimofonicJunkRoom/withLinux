main:clean
	make mmtemplate
	make clean
clean:
	-rm *.log *.toc *.aux *.out
purge:clean
	-rm *.pdf
mmtemplate:
	pdflatex mmtemplate.tex
	pdflatex mmtemplate.tex
b:
	c99 -Wall -O2 -fopenmp b.c -o b
bint:
	c99 -Wall -O2 -fopenmp bint.c -o bint
