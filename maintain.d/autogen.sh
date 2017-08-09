#!/bin/sh
set +e

## automatically generate html pages from your markdown files

# globbing markdown files
project=withLinux
markdownfiles=$(find -L $project -type f -name '*.md')
#echo $markdownfiles

# for each markdown file
for MD in $markdownfiles; do
	echo " => Processing $MD"
	basename=$(basename $MD)
	dirname=$(dirname $MD)
	htmloutput=$(pandoc -f markdown -t html < $MD)
	#echo $htmloutput

	mkdir -p autogen/$dirname
	cp index.css autogen/$dirname
	perl -p \
		-e "s{#STUFF#}{$htmloutput}g;" \
		-e "s{#TITLE_PART2#}{withlinux}g;" \
		-e "s{#PATH#}{$MD}g;" \
		< template.html > autogen/$dirname/$basename.html
done
