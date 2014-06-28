#!/bin/bash

# decode.sh 
# depends on : a8freq a8lu

# Author : C.D.Luminate
#	https://github.com/CDLuminate/a8freq
# changes : created 2014/06/28 
# LICENCE : MIT

# Usage : $0 <SOURCE_FILE> <CODE_FILE>
# decode CODE_FILE using alphabets' freqency in freqency source(SOURCE_FILE).

if [ ! -e ./a8freq ]; then {
	echo "a8freq : error : NOT exist";
	exit;
} fi
if [ ! -e ./a8lu ]; then {
	echo "a8lu : error : NOT exist";
	exit;
} fi

if [ -z "$1" ]; then {
	echo "Usage : $0 <SOURCE_FILE> <CODE_FILE>\ndecode CODE_FILE using alphabets' freqency in SOURCE_FILE.";
	exit;
} fi

SOURCE_FILE=$1
CODE_FILE=$2

if [ ! -e $SOURCE_FILE ]; then {
	echo "$SOURCE_FILE : error : file specified not exist";
	exit;
} fi
if [ ! -e $CODE_FILE ]; then {
	echo "$CODE_FILE : error : file specified not exist";
	exit;
} fi

# Generate source character list
SOURCE_FREQ=$(cat $SOURCE_FILE | ./a8freq -s | head -n26 | sort -n | awk '{print $3}' | tr -d ' \n')
SOURCE_FREQ+=$(echo $SOURCE_FREQ | ./a8lu -r)
echo source_freq_list = $SOURCE_FREQ

# Generate target character list
CODE_FREQ=$(cat $CODE_FILE | ./a8freq -s | head -n26 | sort -n | awk '{print $3}' | tr -d ' \n')
CODE_FREQ+=$(echo $CODE_FREQ | ./a8lu -r)
echo code_freq_list = $CODE_FREQ

# convert
cat $CODE_FILE | tr -s $CODE_FREQ $SOURCE_FREQ > ${CODE_FILE}.new
echo "Converted, see file ./${CODE_FILE}.new"
