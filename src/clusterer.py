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
        print('Distance Matrix:', distance_file)
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
            print('Using CPU to compute HDBSCAN. This may take a while...')
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
    print('Evaluating clusters...')
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
    
    # if len(clust_labs) == n_clusters or n_clusters < 2:
    if n_clusters < 2:
        results.update({'sl_score': None, 'ch_score': None, 'db_score': None, 'entropy': None})
    
    else:
        if method != 'km':
            results.update({
                'sl_score': cython_silhouette_score(clust_data, clust_labs, metric=distance),
                'ch_score': calinski_harabasz_score(clust_data, clust_labs),
                'db_score': davies_bouldin_score(clust_data, clust_labs),
                'entropy' : cython_entropy(clust_labs.astype(np.int32))
            })
        else:
            results.update({
                'sl_score': cython_silhouette_score(clust_data, clust_labs, metric=distance),
                'ch_score': calinski_harabasz_score(clust_data, clust_labs),
                'db_score': davies_bouldin_score(clust_data, clust_labs),
                'entropy' : None
            })

    return results


def get_results(data, filename, labels, dataset, distance, method, **kwargs):
    if method=='km':
        n_clusters = kwargs['n_clusters']
        cls_res = evaluate_cluster(data, labels, distance, method, n_clusters)
    elif method=='dbscan':
        cls_res = evaluate_cluster(data, labels, distance, method)
        cls_res = {'epsilon': kwargs['epsilon'], 'min_pts': kwargs['min_pts']}  | cls_res
    elif method=='hdbscan':
        cls_res = evaluate_cluster(data, labels, distance, method)
        cls_res = {'min_clt_size': kwargs['min_clt_size'], 'min_samples': kwargs['min_samples']}  | cls_res
    
    results = {'distance': distance, 'dataset': dataset} | cls_res
    export_results(Path(data_dir, filename + '.csv'), results)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument('-p', '--path', required=True, type=str, help='File path')
    ap.add_argument('-m', '--method', default='km', type=str, help='Distance metric. Options are "km", "dbscan", "hdbscan".')
    ap.add_argument('-d', '--distance', default='euclidean', type=str, help='Distance metric. Options are "euclidean", "cosine", "wmd".')
    ap.add_argument('-k', '--nclusters', type=int, help='Number of clusters. For KMeans')
    ap.add_argument('-e', '--epsilon', type=float, help='Value of Epsilon. For DBSCAN.')
    ap.add_argument('-M', '--minpts', type=int, help='Minimum points. For DBSCAN.')
    ap.add_argument('-C', '--mcs', type=int, help='Minimum cluster size. For HDBSCAN.')
    ap.add_argument('-S', '--minsample', type=int, help='Minimum sample size min range. For HDBSCAN.')
    
    args = ap.parse_args()
    
    filename = f'eval_{args.method}_{int(datetime.today().timestamp())}'

    model = Doc2Vec.load(str(args.path))
    data = model.dv.vectors
    dataset = '_'.join(Path(args.path).stem.split('_')[1:])

    print(f'Performing {args.method} and evaluating for {dataset} points')
    t0 = datetime.now()

    if args.method=='km':
        labels = get_kmeans(data, dataset, args.distance, args.nclusters)
        get_results(
            data, filename, labels, dataset, args.distance,
            args.method, n_clusters=args.nclusters)
    elif args.method=='dbscan':
        labels = get_dbscan(data, dataset, args.distance, args.epsilon, args.minpts)
        get_results(
            data, filename, labels, dataset, args.distance,
            args.method, epsilon=args.epsilon, min_pts=args.minpts)
    elif args.method=='hdbscan':
        labels = get_hdbscan(data, dataset, args.distance, args.mcs, args.minsample)
        get_results(
            data, filename, labels, dataset, args.distance,
            args.method, min_clt_size=args.mcs, min_samples=args.minsample)
    
    t1 = datetime.now()
    print(f'Took: {t1-t0}')


data_dir = Path('data/interim')
labels_dir = Path(data_dir, 'labels_test')
models_dir = Path('models')

main()
# print([child for child in data_dir.iterdir()])
# print([model for model in models_dir.iterdir()])
