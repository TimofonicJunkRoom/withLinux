dwm:
	keyrate || xset r rate 170 96 # dwm keyrate
	feh --bg-scale $(BGIMG) || xsetroot -solid black # dwm background
	while true; do dwmstatus; done & # dwm status bar
	dunst & # dwm notification
	xbindkeys  # dwm shortcuts
	clipmenud & # dwm clipboard
	ps aux | grep '.*sh.*while true'
