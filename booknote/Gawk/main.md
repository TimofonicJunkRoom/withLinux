Gawk Note
===
> GNU Awk Manual :  
> GAWK: Effective AWK Programming  
> A User’s Guide for GNU Awk  
> Edition 4.1 August, 2016  

# Part I: the awk language

## getting started
```
pattern { action }

awk 'AWKPROGRAM' input.txt
awk -f program.awk input.txt
#!/bin/awk -f
```

```
awk '/pattern/{print $0}' input.txt
awk 'length($0)>80' data.txt
awk '{if (length($0)>max) max = length($0) } END { print max }' data.txt
awk 'NF>0' data.txt # field number > 0, blank lines will be filtered.
awk 'BEGIN { for (i = 1; i <= 7; i++) print int(101 * rand()) }'
ls -l files | awk '{ x += $5 } END { print "total bytes: " x }'
ls -l files | awk '{ x += $5 } END { print "total K-bytes:", x / 1024 }'
awk -F: '{ print $1 }' /etc/passwd | sort
awk 'END { print NR }' data
awk 'NR % 2 == 0' data
awk '/pattern1/{print $0}; /pattern2/{print $0}' data.txt
```

hint, convert TABs into spaces with tool `expand`.

