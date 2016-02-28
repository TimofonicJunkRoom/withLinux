#!/bin/sh
set -e

url_base64="url_coco.base64"
URL=$(base64 -d ${url_base64})

if [ -e "captions_train-val2014.zip" ]; then
	printf "file captions_train-val2014.zip exists. skipping ...\n"
else
	printf "downloading %s ...\n" $URL
	wget -c --progress=dot ${URL}
fi
