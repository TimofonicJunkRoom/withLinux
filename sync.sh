#!/bin/sh
# Simple Debian Source Syncer
# Lumin <cdluminate@163.com>
# BSD-2-Clause
set -e

# core program
RSYNC=/usr/bin/rsync

# parameter
RSYNC_ARG="-4avH -h --del --stats --partial --progress"
SRC="xdlinux.info::debian/"
DST="./debian/"
EXCLUDE="./exclude.txt"
LOG="./debian.log"

# do check first
printf "I: Checking Debian Archive Directory ... "
if [ -d debian ]; then
  printf "[ OK ]\n"
else
  mkdir debian
  printf "[ Created ]\n"
fi

# start syncing
printf "I: Starting to rsync Debian Source ... \n"
${RSYNC} ${RSYNC_ARG} \
  --exclude-from=${EXCLUDE} \
  --log-file=${LOG} \
  ${SRC} ${DST}

# vim
