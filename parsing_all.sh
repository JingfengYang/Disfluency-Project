#!/bin/bash
SPATH=$(pwd)
while IFS='' read -r line || [[ -n "$line" ]]; do
    touch "./result/$line"
    echo "parsing $line"
    python parsing_all.py $line > "./result/$line"
done < "$1"

mkdir train val test
mv result/sw[23][0-9][0-9][0-9] ./train
mv result/sw4[5-9][0-9][0-9] ./val
mv result/sw4[0-1][0-9][0-9] ./test
