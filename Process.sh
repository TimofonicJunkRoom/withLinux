#!/bin/bash

for FILE in $(ls pool/archive/); do
	python3.4 pool/archive/${FILE}
done

bash _trim_json.sh
