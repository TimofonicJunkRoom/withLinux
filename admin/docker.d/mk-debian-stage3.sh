#!/bin/sh
set -e

sudo true # require superuser

echo "=> Making Debian's Stage3 Tarball"
echo "   Usage: $0 DISTRO MIRROR"

TEMP=$(mktemp -d)
ROOT=$TEMP/root
TIMESTAMP=$(date +%Y%m%d)
NAME="Debian-Stage3-$TIMESTAMP.tgz"

mkdir -p $ROOT

sudo debootstrap $1 $ROOT $2
cd $ROOT; sudo tar zcvf ../$NAME .

sudo rm -rf $ROOT

echo "=> Tarball Available here: $TEMP/$NAME"

# docker import $TEMP/$NAME debian:$1.$TIMESTAMP
