#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 MiguelHeCa <josemiguel@heca.tech>
#
# Distributed under terms of the MIT license.

"""
Nearest Neighbours and their calculation
"""
import numpy as np

from pathlib import Path

from cuml.neighbors import NearestNeighbors
from gensim.models.doc2vec import Doc2Vec
from kneed import KneeLocator


def main():
    data_dir = Path('data/interim')
    model_dir = Path('models')
    nn_dir = Path(data_dir, 'nn')
    paths = sorted([path for path in model_dir.glob('*.model')])

    for path in paths:
        print(f"Attempting to obtain nearest neighbors", path.name)
        nn_file = Path(nn_dir, f"nn_{'_'.join(path.stem.split('_')[1:])}.npy")
        
        model = Doc2Vec.load(str(path))
        data = model.dv.vectors.astype(np.float64)

        ran = range(2,31)

        epsilons = np.empty((len(ran), 2))
        counter = 0
        for n in ran:
            nearest_neighbors = NearestNeighbors(n_neighbors=n, metric='l2')
            neighbors = nearest_neighbors.fit(data)

            distances, indices = neighbors.kneighbors(data)
            distances = np.sort(distances[:, distances.shape[1]-1], axis=0)
            i = np.arange(len(distances))
            knee = KneeLocator(i, distances, S=1, curve='convex', direction='increasing', interp_method='polynomial')
            epsilons[counter][0] = n
            epsilons[counter][1] = distances[knee.knee]
            counter += 1

        np.save(nn_file, epsilons)
    
    print('Done')


main()