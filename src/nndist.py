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

import gc
import pickle as pkl

import numpy as np

from datetime import datetime
from pathlib import Path

from hdbscan import HDBSCAN
from gensim.models.doc2vec import Doc2Vec
from cuml.cluster import HDBSCAN as gpu_HDBSCAN