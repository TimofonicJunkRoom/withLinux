main:
	make m
clear:
	rm -r pool
m:
	make clear
	python3.4 miner.py
