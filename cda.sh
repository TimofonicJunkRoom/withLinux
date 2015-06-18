#!/bin/sh
export TEMPDIR=`mktemp -d`
export SHELL="bash"
export UID=`id -u`
export GID=`id -g`

set -e

archivemount -o ro -o uid=${UID} -o gid=${GID} -o nonempty $@ ${TEMPDIR} 
cd ${TEMPDIR}
export TEMPDIR=`pwd`
${SHELL}
cd /
umount ${TEMPDIR}
rmdir ${TEMPDIR}
