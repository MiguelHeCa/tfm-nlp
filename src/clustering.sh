#! /bin/bash
#
# clustering.sh
# Copyright (C) 2022 MiguelHeCa <josemiguel@heca.tech>
#
# Distributed under terms of the MIT license.
#

# do echo "$p ${1:-km} ${2:-euclidean}: $i $j"

for p in models/*.model
do for i in $(seq 5 1 20)
do for j in $(seq 5 1 20)
do python src/clusterer.py -p $p -m hdbscan -d wmd --mcs $i --minsample $j
done
done
done
