Keyboard configuration
===

# in console

for font
```
sudo dpkg-reconfigure console-setup
```

typematic delay and rate
```
kbdrate -d <delay> -r <rate>

/etc/systemd/system/kbdrate.service

[Unit]
Description=Keyboard repeat rate in tty.

[Service]
Type=simple
RemainAfterExit=yes
StandardInput=tty
StandardOutput=tty
ExecStart=/usr/bin/kbdrate -s -d 450 -r 60
 
[Install]
WantedBy=multi-user.target

```
see https://wiki.archlinux.org/index.php/Keyboard_configuration_in_console  

# Xorg

typematic latency and repeat rate
```
xset r rate 192 64
```
