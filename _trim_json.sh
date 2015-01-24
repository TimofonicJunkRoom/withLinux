#!/bin/bash

set -e

for FILE in $(ls pool/archive/*.p); do
	sed -i -e 's/>\\n/>\n/g' pool/archive/${FILE}
done
