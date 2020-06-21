#!/bin/bash

set -e

TRIALS=${1:-5}

while IFS= read -r line
do
    file=$(echo $line | cut -f1 -d" ")
    type=$(echo $line | cut -f2 -d" ")
    for (( c=1; c<=$TRIALS; c++ ))
    do
        python read-benchmark.py $file --format $type
        # Clear the PageCache, dentries, and inodes
        sudo sync
        sudo sh -c "echo 3 > /proc/sys/vm/drop_caches"
    done
done
