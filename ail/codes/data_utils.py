import random, numpy as np

random.seed(0)
np.random.seed(0)

import math, os


def min_max(features): #! OK
    """Return features (numpy array) after min-max normalization"""
    min_features, max_features = [], []
    new_features = np.zeros(features.shape)
    i = 0
    for dim in range(features.shape[1]):
        max_ft, min_ft = max(features[:, dim]), min(features[:, dim])
        min_features.append(min_ft)
        max_features.append(max_ft)

        if max_ft != 0:
            new_features[:, i] = (features[:, dim] - min_ft) / (max_ft - min_ft)

        i += 1
    return new_features

def create_dist_matrix(features): #! OK
    """Compute and return the L2 distance matrix of features array"""
    print("")
    print("Computing the distances between the pairwise of images in features space")
    n = features.shape[0]
    dist_matrix = np.zeros((n, n))
    for i in range(n - 1):
        for j in range(i + 1, n):
            dist_ij = np.linalg.norm(features[i] - features[j])
            dist_matrix[i, j] = dist_ij
            dist_matrix[j, i] = dist_ij
    return dist_matrix
