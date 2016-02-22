# MS CoCo Dataset Python Downloader

## Usage

* download following 2 files from other places e.g. https://github.com/karpathy/neuraltalk2
```
captions_train2014.json
captions_val2014.json
```

* create resource pool
```
$ mkdir pool
$ ls
. .. cocofetch.py pool
```

* run the downloader
```
$ python3 cocofetch.py captions_train2014.json
crunch, crunch, crunch...

# python3 cocofetch.py captions_val2014.json
crunch, crunch, crunch...
```

Happy hacking!
