#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 MiguelHeCa <josemiguel@heca.tech>
#
# Distributed under terms of the MIT license.

"""
Clustering words with KMeans, DBSCAN or HDBSCAN
"""

import argparse
import csv
import gc
import pickle as pkl

import numpy as np

from collections import Counter
from datetime import datetime
from pathlib import Path

from hdbscan import HDBSCAN
from gensim.models.doc2vec import Doc2Vec
from cuml.cluster import HDBSCAN as gpu_HDBSCAN
from cuml.cluster import DBSCAN, KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
from cuml.metrics.cluster.silhouette_score import cython_silhouette_score
from cuml.metrics.cluster.entropy import cython_entropy


def parse_distance_file(dataset, distance):
    if distance == 'wmd':
        mid_string = '_'.join(dataset.split('_')[:-1])
        size = int(dataset.split('_')[-1])
        if size == 300:
            size = 256
        filename = f'{distance}_dist_{mid_string}_{size}.npy'
    else:
        filename = f'{distance}_dist_{dataset}.npy'
    return filename


def export_results(path, data=None):
    if not path.is_file():
        with open(path, 'w') as file:
            writer = csv.writer(file)
            writer.writerow(data.keys())
            if data is not None:
                writer.writerow(data.values())
    else:
        with open(path, 'a') as file:
            writer = csv.writer(file)
            writer.writerow(data.values())
            

def load_labels(file):
    with open(file, 'rb') as handle:
        labels = pkl.load(handle)
            
    return labels


def export_labels(labels, file):
    with open(file, 'wb') as handle:
        pkl.dump(labels, handle, protocol=pkl.HIGHEST_PROTOCOL)


def get_kmeans(data, dataset, distance, n_clusters):
    labels_file = Path(labels_dir, f'labels_km_{dataset}_{n_clusters:02d}_{distance}.pkl')
    if labels_file.is_file():
        labels = load_labels(labels_file)
    else:
        distance_file = Path(data_dir, distance, parse_distance_file(dataset, distance))
        
        if distance_file.is_file():
            data = np.load(distance_file)
        else:
            print(f'{distance.capitalize()} distance matrix not found.')
        
        km = KMeans(n_clusters=n_clusters)
        km.fit(data)
        labels = km.labels_.tolist()
        export_labels(labels, labels_file)
        
    return labels


def get_dbscan(data, dataset, distance, epsilon, min_pts):
    labels_file = Path(labels_dir, f'labels_dbscan_{dataset}_{epsilon}_{min_pts:02d}_{distance}.pkl')
    if labels_file.is_file():
        labels = load_labels(labels_file)
    else:
        distance_file = Path(data_dir, distance, parse_distance_file(dataset, distance))
        if distance_file.is_file():
            data = np.load(distance_file)
            distance = 'precomputed'
        else:
            print(f'{distance.capitalize()} distance matrix not found.')
        
        db = DBSCAN(eps=epsilon,
                    min_samples=min_pts,
                    metric=distance
                   ).fit(data)
        labels = db.labels_
        export_labels(labels, labels_file)
    
    return labels


def get_hdbscan(data, dataset, distance, min_clt_size, min_samples):
    labels_file = Path(labels_dir, f'labels_hdbscan_{dataset}_{min_clt_size:02d}_{min_samples:02d}_{distance}.pkl')
    if labels_file.is_file():
        labels = load_labels(labels_file)
    else:
        if distance == 'euclidean':
            clusterer = gpu_HDBSCAN(
                min_cluster_size=min_clt_size,
                min_samples=min_samples,
                metric=distance
                ).fit(data)
            labels = clusterer.labels_
        else:
            distance_file = Path(data_dir, distance, parse_distance_file(dataset, distance))
            if distance_file.is_file():
                data = np.load(distance_file)
                distance = 'precomputed'
                clusterer = HDBSCAN(
                    min_cluster_size=min_clt_size,
                    min_samples=min_samples,
                    metric=distance
                    ).fit(data.astype(np.float64))
                labels = clusterer.labels_
            else:
                print(f'{distance.capitalize()} distance matrix not found. Terminating program.')
                exit()

        export_labels(labels, labels_file)
    
    return labels


def evaluate_cluster(data, labels, distance, method, n_clusters=None):
    
    results = {}

    if distance == 'wmd':
        distance = 'euclidean'
    
    if method != 'km':
        count_clust = Counter(labels)
        n_clusters = len([key for key in count_clust.keys() if key != -1])
        results['n_clusters'] = n_clusters

        if -1 in count_clust:
            n_noise = count_clust[-1]
            results['n_noise'] = n_noise

        clust_data = []
        clust_labs = []
        for i, label in enumerate(labels):
            if label != -1:
                clust_data.append(data[i])
                clust_labs.append(labels[i])
    else:
        clust_data = data
        clust_labs = labels
        n_clusters = n_clusters
        results['n_clusters'] = n_clusters
    
    clust_data = np.asarray(clust_data)
    clust_labs = np.asarray(clust_labs)
    
    if n_clusters < 2:
        results.update({'sl_score': None, 'ch_score': None, 'db_score': None, 'entropy': None})
    
    else:
        results.update({
            'sl_score': cython_silhouette_score(clust_data, clust_labs, metric=distance),
            'ch_score': calinski_harabasz_score(clust_data, clust_labs),
            'db_score': davies_bouldin_score(clust_data, clust_labs),
            'entropy' : cython_entropy(clust_labs.astype(np.int32))
        })

    return results


def get_results(data, filename, labels, dataset, distance, method, **kwargs):
    if method=='km':
        n_clusters = kwargs['n_clusters']
        cls_res = evaluate_cluster(data, labels, distance, method, n_clusters)
    elif method=='dbscan':
        cls_res = evaluate_cluster(data, labels, distance, method)
        cls_res = {'epsilon': kwargs['epsilon'], 'min_pts': kwargs['min_pts'], 'nn': kwargs['nn']}  | cls_res
    elif method=='hdbscan':
        cls_res = evaluate_cluster(data, labels, distance, method)
        cls_res = {'min_clt_size': kwargs['min_clt_size'], 'min_samples': kwargs['min_samples']}  | cls_res
    
    results = {'distance': distance, 'dataset': dataset} | cls_res
    export_results(Path(eval_dir, filename + '.csv'), results)


def main(path, method, distance, **kwargs):
        
    filename = f'eval_{method}_{int(datetime.today().timestamp())}'

    model = Doc2Vec.load(str(path))
    data = model.dv.vectors
    dataset = '_'.join(path.stem.split('_')[1:])

    
    t0 = datetime.now()

    if method=='km':
        for nclust in range(2,51): # range(21,101)
            print(f'Performing {method} and evaluating for {dataset} points with {nclust} clusters')
            labels = get_kmeans(data, dataset, distance, nclust)
            get_results(data, filename, labels, dataset, distance, method, n_clusters=nclust)
            
    elif method=='dbscan':
        nn_file = Path(nn_dir, f"nn_{'_'.join(path.stem.split('_')[1:])}.npy")
        eps_range = np.load(nn_file)
        # eps_range = [round(e*0.01,3) for e in range(20,101)]
        min_pts_range = [min_pts for min_pts in range(2,16)]
        for nn, epsilon in eps_range:
            for min_pts in min_pts_range:
                print(f'Performing {method} and evaluating for {dataset} points with epsilon {epsilon} and {min_pts} minimum points')
                labels = get_dbscan(data, dataset, distance, epsilon, min_pts)
                get_results(data, filename, labels, dataset, distance, method, epsilon=epsilon, min_pts=min_pts, nn=nn)
                gc.collect()

    elif method=='hdbscan':
        mcs_range = [mcs for mcs in range(2,10)]
        min_samples_range = [ms for ms in range(2,25)]
        for mcs in mcs_range:
            for minsample in min_samples_range:
                print(f'Performing {method} and evaluating for {dataset} points with {mcs} minimum cluster size and {minsample} minimum sample size')
                labels = get_hdbscan(data, dataset, distance, mcs, minsample)
                get_results(data, filename, labels, dataset, distance, method, min_clt_size=mcs, min_samples=minsample)
                gc.collect()
    
    t1 = datetime.now()
    print(f'Took: {t1-t0}')


data_dir = Path('data/interim')
labels_dir = Path(data_dir, 'labels_2')
eval_dir = Path(data_dir, 'evals_4')
models_dir = Path('models')
nn_dir = Path(data_dir, 'nn')

mod_paths = sorted([mod_path for mod_path in Path(models_dir).glob('d2v*.model')])


start = datetime.now()

# for path in mod_paths:
#     nn_file = Path(nn_dir, f"nn_{'_'.join(path.stem.split('_')[1:])}.npy")
#     epsilons = np.load(nn_file)
#     print(path.stem)
#     print(epsilons)

# KMeans
# for p in mod_paths[2:]:
#     main(p, 'km', 'euclidean')
#     gc.collect()
#     main(p, 'km', 'cosine')
#     gc.collect()
#     main(p, 'km', 'wmd')
#     gc.collect()
    # main(p, 'km', 'l2')
    # gc.collect()

# DBSCAN
for p in mod_paths:
    # main(p, 'dbscan', 'euclidean')
    # gc.collect()
    # main(p, 'dbscan', 'cosine')
    # gc.collect()
    # main(p, 'dbscan', 'wmd')
    # gc.collect()
    main(p, 'dbscan', 'l2')
    gc.collect()

# HDBSCAN
# for p in mod_paths[2:]:
    # main(p, 'hdbscan', 'euclidean')
    # gc.collect()
    # main(p, 'hdbscan', 'cosine')
    # gc.collect()
#     main(p, 'hdbscan', 'wmd')
#     gc.collect()_
    # main(p, 'hdbscan', 'l2')
    # gc.collect()

print(f'Finished all datasets. Took {datetime.now()-start}')
