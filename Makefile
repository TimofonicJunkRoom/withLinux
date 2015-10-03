LOGIN ?= packages@qa.debian.org
SourcesGZ := debian/dists/jessie/main/source/Sources.gz

default:
	./sync.sh
bg:
	nohup ./sync.sh &
	sleep 5
	tailf debian.log
dumpmail:
	@python3 dumpmail.py $(SourcesGZ)
stat:
	@python3 stat.py $(SourcesGZ)
who_is_the_most_energetic_dd:
	make dumpmail | sort | uniq -c | sort -n | tac | nl | tac
	# [Rank] [Package count] [Mail]
	# if you want to query your rank, just append "| grep myself"
rank:
	make dumpmail | sort | uniq -c | sort -n | tac | nl | grep $(LOGIN)
	# [Rank] [Package count] [Mail]
dangerous:
	-rm -rf debian/
	-rm debian.log
	-rm nohup.out
