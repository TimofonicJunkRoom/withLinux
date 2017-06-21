#!/bin/sh
set -e
sed \
 -e 's/,/ /g' \
 -e 's/Iris-virginica/3/g' \
 -e 's/Iris-versicolor/2/g' \
 -e 's/Iris-setosa/1/g' \
 iris.data > iris
