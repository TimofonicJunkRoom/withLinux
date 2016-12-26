#!/bin/sh
set -e -x

CHROOT='Gitlab'

echo "chroot into $CHROOT"

mount --bind /proc proc
mount --bind /sys sys
mount --bind /dev dev

echo " -> please run /opt/gitlab/embedded/bin/runsvdir-start in background"

chroot . /bin/bash

umount -R proc
umount -R sys
umount -R dev
