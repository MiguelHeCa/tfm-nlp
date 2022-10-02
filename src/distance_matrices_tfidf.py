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

import cupy as cp
import numpy as np

from datetime import datetime
from pathlib import Path

from cuml.decomposition import PCA
from cuml.metrics import pairwise_distances as gpu_pairwise_distances
from dist_matrix.cuda_dist_matrix_full import dist_matrix as gpu_dist_matrix
from sklearn.metrics import pairwise_distances


def get_euclidean_distance(path, file):
    if not file.is_file():
        # data = cp.load(path, allow_pickle=True)
        data = np.load(path, allow_pickle=True)
        print('Distance matrix not yet calculated. Attempting to obtain it.')
        distances = gpu_dist_matrix(data, metric='euclidean')
        np.save(file, distances)
        print('Distance Matrix saved')
    else:
        print("Euclidean distance matrix already calculated!")
        distances = np.load(file)
    
    return distances


def get_cosine_distance(path, file):
    if not file.is_file():
        # data = cp.load(path, allow_pickle=True)
        data = np.load(path, allow_pickle=True)
        print('Distance matrix not yet calculated. Attempting to obtain it.')
        distances = gpu_dist_matrix(data, metric='cosine')
        np.save(file, distances)
        print('Distance Matrix saved')
    else:
        print("Cosine distance matrix already calculated!")
        distances = np.load(file)
    
    return distances


def get_wmd_distance(path, file):
    if not file.is_file():
        print('Distance matrix not yet calculated. Attempting to obtain it.')
        # data = cp.load(path, allow_pickle=True)
        data = np.load(path, allow_pickle=True)

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
    
    args = ap.parse_args()

    # dist_dir = Path(Path.cwd().parent, f'data/interim/{args.distance}')
    dist_dir = Path(f'data/interim/{args.distance}')
    filename = Path(args.path).name
    dist_file = Path(
        dist_dir,
         'tfidf',
          f"tfidf_{args.distance}_dist_{'_'.join(Path(args.path).stem.split('_')[1:])}.npy"
          )

    print(dist_file)

    print(f"Attempting to obtain {args.distance.capitalize()} distance matrix from", filename)

    if args.distance == 'euclidean':
        distances = get_euclidean_distance(args.path, dist_file)
    elif args.distance == 'cosine':
        distances = get_cosine_distance(args.path, dist_file)
    else:
        distances = get_wmd_distance(args.path, dist_file)

    print(distances)


main()
