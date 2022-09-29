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
from cuml.metrics import pairwise_distances as gpu_pairwise_distances
from dist_matrix.cuda_dist_matrix_full import dist_matrix as gpu_dist_matrix
from gensim.models.doc2vec import Doc2Vec
from sklearn.metrics import pairwise_distances


def get_euclidean_distance(path, file, gpu):
    if not file.is_file():
        model = Doc2Vec.load(path)
        data = model.dv.vectors.astype(np.float64)
        print('Distance matrix not yet calculated. Attempting to obtain it.')
        if gpu == 1:
            distances = gpu_pairwise_distances(data, metric='euclidean')
        else:
            distances = pairwise_distances(data, metric='euclidean')
        np.save(file, distances)
        print('Distance Matrix saved')
    else:
        print("Euclidean distance matrix already calculated!")
        distances = np.load(file)
    
    return distances


def get_cosine_distance(path, file, gpu):
    if not file.is_file():
        model = Doc2Vec.load(path)
        data = model.dv.vectors.astype(np.float64)
        print('Distance matrix not yet calculated. Attempting to obtain it.')
        if gpu == 1:
            distances = gpu_pairwise_distances(data, metric='cosine')
        else:
            distances = pairwise_distances(data, metric='cosine')
        np.save(file, distances)
        print('Distance Matrix saved')
    else:
        print("Cosine distance matrix already calculated!")
        distances = np.load(file)
    
    return distances


def get_wmd_distance(path, file, large_dim, ncomponents):
    if not file.is_file():
        print('Distance matrix not yet calculated. Attempting to obtain it.')
        model = Doc2Vec.load(path)
        data = model.dv.vectors.astype(np.float64)
        if large_dim:
            pca_float = PCA(n_components = ncomponents)
            data = pca_float.fit_transform(data)
            print('Variance explained: ')
            print(sum(pca_float.explained_variance_ratio_))

        weights = np.ones(data.shape, dtype=np.float64)
        # weights = np.zeros(data.shape, dtype=np.float64)
        print('Vectors', data.shape)
        print('Weights', weights.shape) 

        start = datetime.now()

        distances = gpu_dist_matrix(data, V=data, U_weights=weights, V_weights=weights, metric='wasserstein')

        print(f'Took {str(datetime.now() - start)}')
        print(distances.shape)      
        np.save(file, distances)
        print('Distance Matrix saved')
    else:
        print("Word Mover's distance matrix already calculated!")
        distances = np.load(file)
    
    return distances


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument('-p', '--path', type=str, help='Model path')
    ap.add_argument(
        '-d', '--distance', default='euclidean', type=str,
        help='Distance metric. Options are "euclidean", "cosine" and "wmd"')
    ap.add_argument('-g', '--gpu', default=1, type=int, help='Calculate with GPU (1) or CPU (0)')

    args = ap.parse_args()

    # dist_dir = Path(Path.cwd().parent, f'data/interim/{args.distance}')
    dist_dir = Path(f'data/interim/{args.distance}')
    filename = Path(args.path).name
    dist_file = Path(dist_dir, f"{args.distance}_dist_{'_'.join(Path(args.path).stem.split('_')[1:])}.npy")
    gpu = args.gpu

    print(dist_file)

    print(f"Attempting to obtain {args.distance.capitalize()} distance matrix from", filename)

    if args.distance == 'euclidean':
        distances = get_euclidean_distance(args.path, dist_file, gpu)
    elif args.distance == 'cosine':
        distances = get_cosine_distance(args.path, dist_file, gpu)
    else:
        large_dim = False
        size = Path(args.path).stem.split('_')[-1]
        if int(size) > 256:
            ncomponents = 256
            dist_file = Path(dist_dir, f"{args.distance}_dist_{'_'.join(filename.split('_')[1:-1])}_{ncomponents}.npy")
            large_dim = True
        else:
            ncomponents = size
            dist_file = Path(dist_dir, f"{args.distance}_dist_{'_'.join(filename.split('_')[1:-1])}_{size}.npy")
        
        distances = get_wmd_distance(args.path, dist_file, large_dim, ncomponents)

    print(distances)


main()
