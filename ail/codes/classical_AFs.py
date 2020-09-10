import numpy as np
import random

random.seed(0)
np.random.seed(0)

import operator, math
from numpy.linalg import norm
import collections
try:
    import cPickle
except:
    import _pickle as cPickle

class oracle_annotation(object):

    def __init__(self, budget, features, paths, dist_matrix, indexes, true_labels, probas_path): #! OK
        self.budget = budget  # budgets for annotation
        self.features = features  # features
        self.dist_matrix = dist_matrix  # features matrix
        self.indexes = list(
            indexes
        )  # indexes among which the oracle will annotate
        self.true_labels = true_labels  # true labels of images
        self.probas_path = probas_path
        self.paths = paths

    def randomly(self): #! OK
        im_selection = sorted( 
            random.sample(self.indexes, self.budget)
        )
        np.random.shuffle(im_selection)
        return im_selection

    def kcenters(self):  # 
        """select nb_idx indices among im_list_idx and return them
           in im_selection (list) : the minimal distance between two selected images is maximized """
        im_selection = list()
        remaining_im = list(np.copy(self.indexes))

        # initialization : random
        first_idx = self.indexes[random.randint(0, len(remaining_im) - 1)]
        im_selection.append(first_idx)
        remaining_im.remove(first_idx)

        # second : le plus loin
        next_idx = remaining_im[
            np.argmax([self.dist_matrix[im_selection[0], idx] for idx in remaining_im])
        ]
        im_selection.append(next_idx)
        remaining_im.remove(next_idx)

        # ! kcenters, repeat the loop for the rest
        for _ in range(self.budget - 2):
            next_idx = remaining_im[
                np.argmax(
                    self.dist_matrix[np.ix_(im_selection, remaining_im)].min(axis=0)
                )
            ]
            im_selection.append(next_idx)
            remaining_im.remove(next_idx)
        return im_selection


    def entropy(self):
        """
            DESCRIPTION:
            function which selects images for manual annotation based on their entropy.
            The higher the entropy of an image, the higher its position in the list.
            The objective is to put uncertain images first

            PARAMETERS:
            dataset_dir - directory which contains dataset related information
            al_budget - int which gives the active learning manual labeling budget

            RETURNS:
            al_train_indices - list of integer indices of images which are labeled manually
        """
        al_train_indices = []  # list of image indexes to keep

        # open the file which contains the training probabilities
        with open(self.probas_path) as f:
            probs = cPickle.load(f)

        # create a dictionary which stores the images indexes in the list and their entropy
        entropy_dict = {}
        for index in range(0, probs.shape[0]): #parcourir toutes les images(scores)
            crt_entropy = 0  # variable for the entropy of the current probabilities
            for dim in range(0, probs.shape[1]):
                crt_prob = probs[index][dim]
                if crt_prob != 0:
                    crt_entropy = crt_entropy - crt_prob * math.log(crt_prob, 2.0)
            entropy_dict[index] = crt_entropy

        # keep the top queries following the entropy ranking
        cnt_queries = 0
        for ranked_index, entr in sorted(entropy_dict.iteritems(), key=lambda (k, v): (v, k), reverse=True):
            if cnt_queries < self.budget:
                al_train_indices.append(ranked_index)
            cnt_queries = cnt_queries + 1

        return al_train_indices


    def max_margin(self):
        """
            PARAMETERS:
            dataset_dir - directory which contains dataset related information
            al_budget - int which gives the active learning manual labeling budget

            RETURNS:
            al_train_indices - list of integer indices of images which are labeled manually
        """
        al_train_indices = []  # list of image indexes to keep
        # open the file which contains the training probabilities

        with open(self.probas_path) as f:
            probs = cPickle.load(f)

        sorted_probs = np.copy(probs)
        sorted_probs.sort(axis = 1)

        # create a dictionary which stores the images indexes in the list and their margin
        margins_dict = {}
        for index in range(0, probs.shape[0]):
            assert(sorted_probs[index][-1] - sorted_probs[index][-2] >= 0)
            margins_dict[index] = sorted_probs[index][-1] - sorted_probs[index][-2]


        # keep the top queries following the margin ranking
        cnt_queries = 0
        for ranked_index, conf in sorted(margins_dict.iteritems(), key=lambda (k, v): (v, k), reverse=True):
            if cnt_queries < self.budget:
                al_train_indices.append(ranked_index)
            cnt_queries = cnt_queries + 1

        return al_train_indices


    def run(self, manual_method):
        dispatcher = {
            'randomly': self.randomly,
            'kcenters': self.kcenters,
            'entropy': self.entropy,
            'max_margin': self.max_margin,
        }

        annotated_indices = eval(manual_method+'()',{'__builtins__':None},dispatcher)
        self.annotated_img_idx = annotated_indices
        self.annotated_features = self.features[self.annotated_img_idx]
        self.annotated_classes = self.true_labels[self.annotated_img_idx]
        self.annotated_paths = self.paths[self.annotated_img_idx]
        self.classes = list(set(self.annotated_classes))


    def update(self, annotated_img_idx, L2_annotated_features, annotated_classes, annotated_paths):
        self.annotated_img_idx = annotated_img_idx
        self.annotated_features = L2_annotated_features
        self.annotated_classes = annotated_classes
        self.annotated_paths = annotated_paths
        self.classes = list(set(self.annotated_classes))
