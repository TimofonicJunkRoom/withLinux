#!/bin/bash


cd pool/archive

for FILE in $(ls); do
	grep "tweet-text" ${FILE} > ../${FILE}.filtered
done
