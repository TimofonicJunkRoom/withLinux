default:
	./sync.sh
bg:
	nohup ./sync.sh &
	tailf debian.log
stat:
	@python3 stat.py debian/dists/jessie/main/source/Sources.gz
who_is_the_most_energetic_dd:
	make stat | sort | uniq -c | sort -n | tac | nl | tac
	# if you want to query your rank, just append "| grep myself"
dangerous:
	-rm -rf debian/
	-rm debian.log
	-rm nohup.out
