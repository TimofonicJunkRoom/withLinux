# MS CoCo Dataset Python Downloader

COCO statistics
```
validation                 = 40504
train                      = 82783
Total = validation + train = 123287
total size                 = 18881856 K
```

## Usage

* download json file, then please extract it.
```
$ bash download_annotation.sh
$ unzip -d . captions_train-val2014.zip
```

* download using single process
```
$ python3 cocofetch.py captions_train2014.json
[crunch, crunch, crunch...]

$ python3 cocofetch.py captions_val2014.json
[crunch, crunch, crunch...]
```

* scan for broken jpegs, delete them
```
$ python3 check_jpeg.py pool/ | tee junk
$ cat junk | awk '{print $3}' | xargs rm # remove troublesome images
```
then you should switch download URL from flickr to mscoco in `cocofetch.py`, and download again. (existing images will be skipped, so this is fast.)

* scan for missing files if any.
```
$ python3 scan_missing XXX.json
```
if it says nothing, no picture is missed.

Happy hacking!

#### trick: multiple process download

first we split json files
```
$ python3 split.py annotations/train.json
```
then start 2 processes to download
```
$ python3 cocofetch.py annotations/train.json.left
$ python3 cocofetch.py annotations/train.json.right
```
Tip: you can split the json file for several times, so you can launch more processes.
```
$ python3 split.py annotations/train.json.left
$ python3 split.py annotations/train.json.right

$ ... cocofetch.py train.json.left.left &
$ ... cocofetch.py train.json.left.right &
$ ... cocofetch.py train.json.right.left &
$ ... cocofetch.py train.json.right.right &
```
