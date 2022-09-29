#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 MiguelHeCa <josemiguel@heca.tech>
#
# Distributed under terms of the MIT license.

"""
Word Mover's Distance Applied to Enron
"""
import argparse

import numpy as np

from datetime import datetime
from pathlib import Path

from cuml.decomposition import PCA
from dist_matrix.cuda_dist_matrix_full import dist_matrix as gpu_dist_matrix
from gensim.models.doc2vec import Doc2Vec


def main():
    wmd_dir = Path(Path.cwd().parent, 'data/interim/wmd')

    ap = argparse.ArgumentParser()

    ap.add_argument('-p', '--path', type=str, help='Model path')

    args = ap.parse_args()

    filename = Path(args.path).name

    print("Attempting to obtain Word Mover's Distance Matrix from", filename)

    large_dim = False
    size = Path(args.path).stem.split('_')[-1]
    if int(size) > 256:
        ncomponents = 256
        wmd_file = Path(wmd_dir, f"wmd_dist_{'_'.join(filename.split('_')[1:-1])}_{ncomponents}.npy")
        large_dim = True
    else:
        wmd_file = Path(wmd_dir, f"wmd_dist_{'_'.join(filename.split('_')[1:-1])}_{size}.npy")

    if not wmd_file.is_file(): 
        print('Distance matrix not yet calculated. Attempting to obtain it.')
        model = Doc2Vec.load(str(Path(args.path)))
        data = model.dv.vectors.astype(np.float64)
        if large_dim:
            pca_float = PCA(n_components = ncomponents)
            data = pca_float.fit_transform(data)
            print('Variance explained: ')
            print(sum(pca_float.explained_variance_ratio_))

        weights = np.ones(data.shape, dtype=np.float64)
        print('Vectors', data.shape)
        print('Weights', weights.shape) 

        start = datetime.now()
        distances = gpu_dist_matrix(data, V=data, U_weights=weights, V_weights=weights, metric='wasserstein')

        print(f'Took {str(datetime.now() - start)}')
        print(distances.shape)

        np.save(wmd_file, distances)
        print('Distance Matrix saved')
    else:
        print("Word Mover's distance matrix already calculated!")
        distances = np.load(wmd_file)
    
    print(distances)


main()
