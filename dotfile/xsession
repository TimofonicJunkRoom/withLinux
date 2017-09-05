# lumin's xsession

xset r rate 190 60
xset -b

xsetroot -solid "#333333"

export _JAVA_AWT_WM_NONREPARENTING=1
xautolock -time 1 -locker "slock" &
make -C $HOME/ dwm &
exec dwm
