# MS CoCo Dataset Python Downloader

## Usage

#### download json file

download following 2 files from other places e.g. https://github.com/karpathy/neuraltalk2
```
captions_train2014.json
captions_val2014.json
```

#### create resource pool
```
$ mkdir pool
$ ls
. .. cocofetch.py pool
```

#### method 1: single process download
run the downloader
```
$ python3 cocofetch.py captions_train2014.json
crunch, crunch, crunch...

# python3 cocofetch.py captions_val2014.json
crunch, crunch, crunch...
```

#### method 2: multiple process download
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

## check for broken jpeg

```
# python3 check_jpeg.py
```

Happy hacking!
