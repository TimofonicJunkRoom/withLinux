#!/bin/sh
set -e

tempfile=$(mktemp)
cat >> $tempfile <<EOF
127.0.0.1 localhost
::1       localhost
8.8.8.8   googledns
192.168.0.1 localnet1
192.168.0.2 localnet2
EOF
trap "rm -f $tempfile" EXIT

# regex

echo no pattern specified
awk '//{print}' $tempfile
echo

echo match localhost
awk '/localhost/{print}' $tempfile
echo

echo using dot wildcard
awk '/l.c/{print}' $tempfile
echo

echo using asterisk wildcard
awk '/l*t/{print}' $tempfile
echo

echo using characters match
awk '/t[12]/{print}' $tempfile
echo

echo using range
awk '/[0-9]/{print}' $tempfile
echo

echo using ^ and $
awk '/^:/{print}' $tempfile
awk '/s$/{print}' $tempfile
echo

echo using escape
awk '/\$/{print}' $tempfile
echo

# fields

echo fields
awk '/localhost/{print $2, $1}' $tempfile
awk '/localhost/{printf "%-10s %s\n", $2, $1}' $tempfile
echo
