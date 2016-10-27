SSH Notes
===

SFTP chroot
-----------
https://wiki.archlinux.org/index.php/SFTP_chroot

SSH could not load host key
---
```shell
ssh-keygen -t dsa -f /etc/ssh/ssh_host_dsa_key
ssh-keygen -t rsa -f /etc/ssh/ssh_host_rsa_key
ssh-keygen -t ecdsa -f /etc/ssh/ssh_host_ecdsa_key
ssh-keygen -t ed25519 -f /etc/ssh/ssh_host_ed25519_key

OR SIMPLY

ssh-keygen -A
```
