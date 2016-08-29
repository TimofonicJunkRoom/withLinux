ZFS
===

installing ZFS with `apt install zfs`

## create and use ZFS on usb stick

```
# stat /dev/sdc
# zpool list -v
no pools available
# zpool create -f luminz sdc
# zpool list -v
NAME   SIZE  ALLOC   FREE  EXPANDSZ   FRAG    CAP  DEDUP  HEALTH  ALTROOT
luminz  7.25G    64K  7.25G         -     0%     0%  1.00x  ONLINE  -
  sdc  7.25G    64K  7.25G         -     0%     0%
# zfs list -tall
NAME     USED  AVAIL  REFER  MOUNTPOINT
luminz    55K  7.02G    19K  /luminz
# zfs snapshot luminz@test
# zfs list -tall
NAME          USED  AVAIL  REFER  MOUNTPOINT
luminz        230K  7.02G    19K  /luminz
luminz@test      0      -    19K  -
# zfs rollback luminz@test
# # zfs create luminz/images
# zfs list -tall
NAME            USED  AVAIL  REFER  MOUNTPOINT
luminz           87K  7.02G    19K  /luminz
luminz@test       1K      -    19K  -
luminz/images    19K  7.02G    19K  /luminz/images
# zpool export luminz
# zpool list -v
no pools available

[ eject usb disk and re-plug it in ]

# zpool import
   pool: luminz
     id: 8904241131819414955
  state: ONLINE
 action: The pool can be imported using its name or numeric identifier.
 config:

	luminz      ONLINE
	  sdc       ONLINE
# zpool import luminz
# zpool list -v
NAME   SIZE  ALLOC   FREE  EXPANDSZ   FRAG    CAP  DEDUP  HEALTH  ALTROOT
luminz  7.25G   270K  7.25G         -     0%     0%  1.00x  ONLINE  -
  sdc  7.25G   270K  7.25G         -     0%     0%
```

Note, the safe way to use USB+ZFS is `zpool import/export`.

## reference

`man zfs`, `man zpool`
