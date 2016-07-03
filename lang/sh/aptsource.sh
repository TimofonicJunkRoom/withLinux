#!/bin/bash
# Copyright (C) 2016 Zhou Mo <cdluminate@gmail.com>
# MIT LICENSE
#
# User-friendly APT sources.list configurator
set -e

# setup tmpfile
tempfile=$(mktemp)
trap "rm -f $tempfile" 0 0 1 2 3 9 15

# open dialog
# --[ select mirror with radiolist
dialog --stdout \
  --title "Select Mirror" \
  --clear "$@" \
  --radiolist "Select a mirror:" \
    20 72 5 \
    "ftp.cn.debian.org" "" on \
    "mirror.tuna.tsinghua.edu.cn" "" off \
    "ftp.xdlinux.info" "" off \
  2>&1 1>$tempfile 
retval=$?
mirror=$(cat $tempfile)
echo "select mirror $mirror"

#EOF
