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

## running awk and gawk
pass

functionality to include awk files is available in awk.

## regular expressions
```
exp ~ /pattern/  # match expression with pattern
exp !~ /pattern/ # not match

$1 ~ /patter/    # field 1 match
```

regular expression operations
```
\      escape
^      string begining
$      string ending
.      match a single character
[...]  matches any one character provided within brackets
|      alternation
(...)  grouping regular expressions, e.g. (apple|banana)
*      repeat preceding character or not, greedy.
+      repeat preceding character at least once
?      repeat once or not.
{n}
{n,}
{n,m}  interval expression
```
see also: POSIX character classes

case sensitivity
```
tolower($1) ~ /regexp/
```

## chap4: reading input files
```
FILENAME
RS = "u" # record separator
NF       # number of fields
NR       # number of records read so far, starts from 1
FS       # field separator
BEGIN { FIELDWIDTHS = "9 6 10 6 7 7 35" }
```

`getline` function ...

## chap5: printing output
```
print item1, item2, ...
print "this is a string\n"
print $1 $2   # <-- this is an common error, which yields no space between items
print $1, $2  # correct.

msg = "do not panic!"
printf "%s\n", msg
```

`OFS` is output field separator.

output redirectoring # seems important to me.
```
$ awk ’{ print $2 > "phone-list"
>        print $3 >> "append"
>        print $4 | "sort -r > c4.sorted"
>        print $1 > "name-list" }’ mail-list
```

## chap6: expressions

FIXME
FIXME
FIXME
FIXME
FIXME
FIXME
FIXME
FIXME
FIXME
FIXME
FIXME
FIXME
FIXME
FIXME
FIXME
FIXME
FIXME
FIXME
FIXME

## chap9: functions
