main:
	make clear
	make parse_list
	python3.4 pullmsg.py
clear:
	rm -r pool
gen:
	bash gen.sh
parse_list:
	bash parse_list.sh
m:
	make clear
	python3.4 miner.py
