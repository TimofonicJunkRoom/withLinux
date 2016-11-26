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
```
see https://wiki.archlinux.org/index.php/Keyboard_configuration_in_console  

# Xorg

typematic latency and repeat rate
```
xset r rate 192 64
```
