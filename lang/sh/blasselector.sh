#!/bin/bash
# Copyright (C) 2016 Zhou Mo <cdluminate@gmail.com>
# MIT LICENSE
#
# User-friendly BLAS selector for Debian/Ubuntu
set -e

# check requirements
if ! test "root" = $(whoami); then
  echo Run this script as root.
  exit 1
fi

# setup tmpfile
tempfile=$(mktemp)
trap "rm -f $tempfile" 0 0 1 2 3 9 15

# open dialog
dialog \
  --stdout --clear \
  --backtitle "Lumin" \
  --title "BLAS Selector for Debian/Ubuntu" \
  --clear "$@" \
  --radiolist "Please select a BLAS library, with the SPACE key." \
    17 72 5 \
    "OpenBLAS" "optimized BLAS library (Recommend)" on \
    "Atlas" "Automatically Tuned Linear Algebra Software" off \
    "Generic" "Basic Linear Algebra Reference implementation" off \
  2>&1 1>$tempfile

retval=$?
BLAS=$(cat $tempfile)
case $retval in
  0) #OK
    echo "Select $BLAS ." ;;
  *) #Cancel
    echo "Abort."; exit 3 ;;
esac

# work
if test $BLAS = "OpenBLAS"; then
  apt install libopenblas-base libopenblas-dev
  update-alternatives --set libblas.so   /usr/lib/openblas-base/libblas.so
  update-alternatives --set libblas.so.3 /usr/lib/openblas-base/libblas.so.3
elif test $BLAS = "Atlas"; then
  apt install libatlas-base libatlas-dev
  update-alternatives --set libblas.so   /usr/lib/atlas-base/atlas/libblas.so
  update-alternatives --set libblas.so.3 /usr/lib/atlas-base/atlas/libblas.so.3
elif test $BLAS = "Generic"; then
  apt install libblas3 libblas-dev
  update-alternatives --set libblas.so   /usr/lib/libblas/libblas.so
  update-alternatives --set libblas.so.3 /usr/lib/libblas/libblas.so.3
else
  echo Unknown Error.; exit 4
fi
