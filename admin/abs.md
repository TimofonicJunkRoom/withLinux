Arch Build System
===

# example: rebuild dwm

```
sudo pacman -S base-devel
sudo pacman -S abs

sudo abs community/dwm
cp -av /var/abs/community/dwm ~/packages
cd ~/packages/dwm
vim config.h
vim PKGBUILD # change hashsum
makepkg -f
sudo pacman -U dwm-6.1-3.xxx.tar.gz
```

abs config file: `/etc/abs.conf`

reference: 
https://wiki.archlinux.org/index.php/Arch_Build_System
https://wiki.archlinux.org/index.php/Pacman
https://wiki.archlinux.org/index.php/PKGBUILD
