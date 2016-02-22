# MS CoCo Dataset Python Downloader

## Usage

1. download following 2 files from other places e.g. https://github.com/karpathy/neuraltalk2
```
captions_train2014.json
captions_val2014.json
```

2. create resource pool
```
$ mkdir pool
$ ls
. .. cocofetch.py pool
```

3. run the downloader
```
$ python3 cocofetch.py captions_train2014.json
crunch, crunch, crunch...

# python3 cocofetch.py captions_val2014.json
crunch, crunch, crunch...
```

Happy hacking!
