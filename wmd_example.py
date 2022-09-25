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
# import pandas as pd

from datetime import datetime
from pathlib import Path

from cuml.decomposition import PCA
from dist_matrix.cuda_dist_matrix_full import dist_matrix as gpu_dist_matrix
from gensim.models.doc2vec import Doc2Vec


def main():
    # models_dir = Path(Path.cwd().parent, 'models')
    ap = argparse.ArgumentParser()

    ap.add_argument('-p', '--path', type=str, help='Size of features or columns')


    # ap.add_argument('-f', '--features', type=int,
    #                 default=10, help='Size of features or columns')
    # ap.add_argument('-r', '--rows', type=int,
    #                 default=100, help='Size of rows')

    args = ap.parse_args()

    model = Doc2Vec.load(str(Path(args.path)))
    data = model.dv.vectors.astype(np.float64)

    if data.shape[1] > 256:
        pca_float = PCA(n_components = 256)
        data = pca_float.fit_transform(data)
        print(sum(pca_float.explained_variance_ratio_))

    weights = np.ones(data.shape, dtype=np.float64)
    print('Vectors', data.shape)
    print('Weights', weights.shape)
 
    # n_features = args.features
    # n_rows = args.rows
    # U, V, U_weights, V_weights = np.random.rand(4, args.rows, args.features)
    # print('U: ', U.shape)
    # print('V: ', V.shape)
    

    start = datetime.now()
    # distances = gpu_dist_matrix(U=U, V=V, U_weights=U_weights, V_weights=V_weights, metric='wasserstein')
    distances = gpu_dist_matrix(data, V=data, U_weights=weights, V_weights=weights, metric='wasserstein')

    print(f'Took {str(datetime.now() - start)}')
    print(distances.shape)
    print(distances)


main()
