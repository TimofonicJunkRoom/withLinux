default:
	./sync.sh
bg:
	nohup ./sync.sh &
	tailf debian.log
dangerous:
	-rm -rf debian/
	-rm debian.log
	-rm nohup.out
