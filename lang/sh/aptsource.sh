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
dialog --stdout --no-lines --no-shadow \
  --title "Select Mirror" \
  --clear "$@" \
  --radiolist "Select a mirror:" \
    16 72 0 \
    "http://ftp.cn.debian.org" "" on \
    "http://mirror.tuna.tsinghua.edu.cn" "" off \
    "http://ftp.xdlinux.info" "" off \
    "file:///media/lumin/Seagate/DebArchive.git/jessie/" "" off \
  2>&1 1>$tempfile 
retval=$?
mirror=$(cat $tempfile)
echo "select mirror $mirror"

# --[ select release with radiolist
dialog --stdout --no-lines --no-shadow \
  --title "Select Release" "$@" \
  --checklist "Select a release:" \
    16 72 10 \
    "jessie" "" on \
    "jessie-backports" "" on \
    "jessie-proposed-updates" "" on \
    "jessie/updates" "" on \
    "sid" "" off \
    "experimental" "" off \
    "corresponding-source" "" off \
  2>&1 1>$tempfile 
retval=$?
releases=$(cat $tempfile)
echo "select release $releases"

#deb http://debug.mirrors.debian.org/debian-debug unstable-debug main

#EOF
