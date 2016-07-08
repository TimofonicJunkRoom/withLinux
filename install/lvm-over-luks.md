LVM over LUKS
---

Debian Installer, expert install method supports it.  
"guided partition with LVM and encryption".
It is also easy to manually set LVM over luks with that installer.

There is no notable difference for `fstab`:
```
/dev/mapper/debian-vg-root / ext4 errors=remount-ro 0 1
UUID=<UUID> /boot ext2 defaults 0 2
/dev/mapper/debian-vg-home /home ext4 defaults 0 2
/dev/mapper/debian-vg-swap none swap sw 0 0
/dev/sr0 /media/cdrom0 udf,iso9660 user,noauto 0 0
```
and there is no notable difference in grub config, either.
```
linux /vmlinuz root=/dev/mapper/debian-vg-root ro quiet
```
but note something should appear in file `/etc/crypttab`, e.g.
```
cdisk0 UUID=12345678-9abc-def012345-6789abcdef01 none luks
```

LVM over LUKS for ArchLinux
---
> https://wiki.archlinux.org/index.php/Dm-crypt/Encrypting_an_entire_system  
> https://wiki.archlinux.org/index.php/Installation_guide  

Experiment performed on ArchLinux with Virtualbox.

* partition disk

GPT partition table
```
1mb-2mb bios_grub  -> sda1 (toggle bios_grub on)
2mb-256mb boot     -> sda2
256mb-100% luks    -> sda3
```

* make luks parts
```
cryptsetup luksFormat /dev/sda3
cryptsetup luksOpen /dev/sda3 luks
```

* setup lvm over luks
```
lvm pvcreate /dev/mapper/luks
lvm vgcreate lvm /dev/mapper/luks
lvm lvcreate -L 512M lvm -n swap
lvm lvcreate -l 100%FREE lvm -n root
```

* setup file system
```
mkfs.ext4 /dev/sda2
mkfs.ext4 /dev/mapper/lvm-root
mkswap /dev/mapper/lvm-swap
```

* mount for installation
```
mount /dev/mapper/lvm-root /mnt
swapon /dev/mapper/lvm-swap
mkdir /mnt/boot
mount /dev/sda2 /mnt/boot
```

* add hook in `/mnt/etc/mkinitcpio.conf`
```
HOOK=" ... encrypt lvm2 ... filesystems ... "
```

* install base system: edit `/etc/pacman.d/mirrorlist`
```
pacstrap /mnt base vim openssh
genfstab -p /mnt > /mnt/etc/fstab
vim /mnt/etc/hostname
arch-chroot /mnt
mkinitcpio -p linux # note: you should see [encrypt] and [lvm2] here, or it goes wrong.
systemctl enable dhcpcd@enp0s3.service
passwd
```

* install bootloader
```
grub-install --boot-directory=/mnt/boot/ /dev/sda
arch-chroot /mnt/
grub-mkconfig -o /boot/grub/grub.cfg
```
edit the config?
```
cryptdevice=UUID=<device_uuid>:lvm root=/dev/mapper/lvm-root
OR
cryptdevice=UUID=<device_uuid>:lvm root=UUID=<UUID>
```

* reboot

* post installation
```
pacman -Ss ecryptfs
pacman -S ecryptfs-utils
ecryptfs-setup-private || modprobe ecryptfs
ecryptfs-setup-private || modprobe ecryptfs
ecryptfs-mount-private
ecryptfs-umount-private
```
