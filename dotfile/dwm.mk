dwm:
	keyrate || xset r rate 170 96 # dwm keyrate
	feh --bg-scale $(BGIMG) || xsetroot -solid black # dwm background
	while true; do dwmstatus; done & # dwm status bar
	killall dunst; dunst & # dwm notification
	killall xbindkeys; xbindkeys  # dwm shortcuts
	killall clipmenud; clipmenud & # dwm clipboard
	#xcompmgr & # transparent bar patch
	ps aux | grep '.*sh.*while true'
