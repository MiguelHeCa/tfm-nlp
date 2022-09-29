#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 MiguelHeCa <josemiguel@heca.tech>
#
# Distributed under terms of the MIT license.

"""
Cosine Distance Calculator
"""
import argparse

import numpy as np

from collections import Counter
from datetime import datetime
from pathlib import Path

from gensim.models.doc2vec import Doc2Vec
from cuml.metrics import pairwise_distances

def main():
    cosine_dir = Path(Path.cwd().parent, 'data/interim/cosine')

    ap = argparse.ArgumentParser()

    ap.add_argument('-p', '--path', type=str, help='Model path')

    args = ap.parse_args()

    filename = Path(args.path).name

    print("Attempting to obtain Cosine Distance Matrix from", filename)

    wmd_file = Path(cosine_dir, f"cosine_dist_{'_'.join(Path(args.path).stem.split('_')[1:])}.npy")

    if not cosine_dir.is_file(): 
        print('Distance matrix not yet calculated. Attempting to obtain it.')
        model = Doc2Vec.load(str(Path(args.path)))
        data = model.dv.vectors.astype(np.float64)

        start = datetime.now()
        distances = pairwise_distances(data, metrix='cosine')
        print(f'Took {str(datetime.now() - start)}')
        print(distances.shape)

        

        np.save(wmd_file, distances)
        print('Distance Matrix saved')
    else:
        print("Cosine Distance matrix already calculated!")
        distances = np.load(wmd_file)
    
    print(distances)


main()
