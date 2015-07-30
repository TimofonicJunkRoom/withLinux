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
