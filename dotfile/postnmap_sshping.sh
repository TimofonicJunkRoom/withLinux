#!/bin/sh
set +e

# 1. nohup nmap -p22 -T5 10.170.0.0/16 &
# 2. bash <THIS_SCRIPT> nohup.out

if test -z "$1"; then
	printf "NO NMAP LOG FOUND\n"
fi

iplist=$(cat $1 | grep open -B3 | grep ^Nmap | awk '{print $5}')
printf "\x1b[31mIPLIST\x1b[m\n"
echo $iplist
printf "\x1b[31mIPLIST\x1b[m\n"

for ip in $iplist; do
	printf "\x1b[33m$ip\x1b[m :\t";
   	curl -m1 -s $ip:22;
        continue
done
