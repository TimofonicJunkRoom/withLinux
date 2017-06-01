dwm:
	keyrate || xset r rate 170 96 # dwm keyrate
	feh --bg-scale $(BGIMG) || xsetroot -solid gray # dwm background

	-pkill -f -u $$USER '.*/bin/sh.*-c.*while true.*dwmstatus.*done'
	while true; do dwmstatus; done & # dwm status bar

	-pkill -u $$USER dunst;
	dunst & # dwm notification

	-pkill -u $$USER xbindkeys;
	xbindkeys  # dwm shortcuts

	-pkill -u $$USER clipmenud;
	clipmenud & # dwm clipboard

	-pkill -u $$USER xcompmgr;
	xcompmgr & # transparent bar patch
