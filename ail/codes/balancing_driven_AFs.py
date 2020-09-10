#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import division

import  os
import copy

import numpy as np

import math
import random
random.seed(0)
import time
import datetime
import sys
from data_utils import create_dist_matrix




def poor(candidate_features, candidate_labels, candidate_paths, al_budget, annotated_file_path, annotated_features_file_path):
	"""
	DESCRIPTION:
	balancing method which selects the points which maximize the difference between classes having the least number of images and the maximum score of major classes

	PARAMETERS:
	-al_train_indices - indices selected by the "pure" active learning method
	-dataset_dir - directory for the current dataset
	-al_budget - budget for manual annotation with active learning
	-switch - position in AL list after which balancing is performed instead of AL

	RETURNS:
	selected - balanced version of the training indices. In this case, a copy of the input
	"""

	annotated_file = open(annotated_file_path, 'r').readlines()
	annotated_features_file = open(annotated_features_file_path, 'r').readlines()
	assert(len(annotated_file) == len(annotated_features_file))

	### get class -> features
	label_features_dict = {}
	label_mean_feat_dict = {}
	label_count_dict = {}
	for line_path, line_feat in zip(annotated_file, annotated_features_file):

		#paths
		line_path = line_path.strip().split()
		path, label = line_path[0], int(line_path[1])

		##features:
		line_feat = line_feat.strip().split()
		features = np.array(line_feat, dtype=np.float)

		if label not in label_count_dict.keys():
			label_count_dict[label] = 1
			label_features_dict[label] = features
		else:
			label_count_dict[label] += 1
			label_features_dict[label] = np.vstack((label_features_dict[label], features))


	classes = sorted(set(label_count_dict.keys())) #classes that i already discovered
	print('Previously discovered labels= ' + str(classes))


	#target : new images that i will annotate
	labels = None
	paths = None
	features = None

	#  read train list and store labels in a list
	print("{} classes = {}".format(len(classes), str(classes)))
	print "loaded embeddings:", candidate_features.shape

	print('before')
	print(str(label_count_dict))
	##################go to balancing mode ###################################################

	print "start balancing:"

	class_matched = 0
	class_mismatched = 0
	selected = 0

	minors = []
	majors = []
	mean_count = math.floor(np.mean(np.array([label_count_dict[k] for k in label_count_dict.keys()])))

	assert(sorted(label_count_dict.keys()) == sorted(label_features_dict.keys()))

	for label in label_count_dict.keys():

		if label_count_dict[label] < mean_count:
			minors.append(label)
		else:
			majors.append(label)

		#mean features
		label_mean_feat_dict[label] = np.mean(label_features_dict[label], axis=0)


	while selected < al_budget:
		# print "selected = {}, remaining = {}".format(len(selected), len(remaining))
		rarest_count = np.Inf
		# get the label with minimum numbner of images
		for label in label_count_dict.keys():
			if label_count_dict[label] < rarest_count:
				rarest_count = label_count_dict[label]

		# updated_labels = []

		for label in label_count_dict.keys():
			if label_count_dict[label] == rarest_count:
				min_minus_max_majors = []
				for feature in candidate_features:
					dist_label = np.linalg.norm(feature - label_mean_feat_dict[label])
					assert(dist_label >= 0)
					min_dist_maj_label = min([np.linalg.norm(feature - label_mean_feat_dict[maj_label]) for maj_label in majors])
					min_minus_max_majors.append(dist_label - min_dist_maj_label)

				best_idx = np.argmin(np.array(min_minus_max_majors))
				if selected < al_budget :
					# get the true class of the selected index to simulate a human annotation
					true_label = candidate_labels[best_idx]
					true_path = candidate_paths[best_idx]
					true_feature = candidate_features[best_idx]
					# updated_labels.append(true_label)

					if true_label == label:
						class_matched = class_matched + 1
					else:
						class_mismatched = class_mismatched + 1

					selected += 1

					if paths is None:
						labels = true_label
						paths = true_path
						features = true_feature
					else:
						labels = np.append(labels, true_label)
						paths = np.append(paths, true_path)
						features = np.vstack((features, true_feature))

					if not true_label in label_count_dict.keys():
						label_count_dict[true_label] = 1
						label_features_dict[true_label] = true_feature
					else:
						label_count_dict[true_label] += 1
						label_features_dict[true_label] = np.vstack((label_features_dict[true_label], true_feature))

					label_mean_feat_dict[true_label] = np.mean(label_features_dict[true_label], axis=0)

					candidate_features = np.delete(candidate_features, best_idx, axis=0)
					candidate_labels = np.delete(candidate_labels, best_idx, axis=0)
					candidate_paths = np.delete(candidate_paths, best_idx, axis=0)

					minors = []
					majors = []
					mean_count = math.floor(np.mean(np.array([label_count_dict[k] for k in label_count_dict.keys()])))

					for label in label_count_dict.keys():
						if label_count_dict[label] < mean_count:
							minors.append(label)
						else:
							majors.append(label)



					#######################################

				elif selected >= al_budget:
					break
				else:
					print('error')


	print('after')
	print(str(label_count_dict))

	print "balancing class matches and mismatches:", class_matched, class_mismatched
	print "stopped balancing:"
	assert (selected == len(labels) == len(paths))
	return labels, paths, features


###############
def bcore(dist_matrix, candidate_features, candidate_labels,
					  candidate_paths, al_budget, annotated_file_path, annotated_features_file_path):
	"""
	DESCRIPTION:
	balancing method which selects the points which maximize the difference between classes having the least number of images and the maximum score of major classes

	PARAMETERS:
	-al_train_indices - indices selected by the "pure" active learning method
	-dataset_dir - directory for the current dataset
	-al_budget - budget for manual annotation with active learning
	-switch - position in AL list after which balancing is performed instead of AL

	RETURNS:
	selected - balanced version of the training indices. In this case, a copy of the input
	"""

	annotated_file = open(annotated_file_path, 'r').readlines()
	annotated_features_file = open(annotated_features_file_path, 'r').readlines()
	assert(len(annotated_file) == len(annotated_features_file))

	### get class -> features
	label_features_dict = {}
	label_mean_feat_dict = {}
	label_count_dict = {}
	for line_path, line_feat in zip(annotated_file, annotated_features_file):
		#paths
		line_path = line_path.strip().split()
		path, label = line_path[0], int(line_path[1])

		##features:
		line_feat = line_feat.strip().split()
		features = np.array(line_feat, dtype=np.float)

		if label not in label_count_dict.keys():
			label_count_dict[label] = 1
			label_features_dict[label] = features
		else:
			label_count_dict[label] += 1
			label_features_dict[label] = np.vstack((label_features_dict[label], features))


	classes = sorted(set(label_count_dict.keys())) #classes that i already discovered
	print('Previously discovered labels= ' + str(classes))

	#  read train list and store labels in a list
	print("{} classes = {}".format(len(classes), str(classes)))
	print "loaded embeddings:", candidate_features.shape

	print('before')
	print(str(label_count_dict))
	##################go to balancing mode ###################################################

	print "start balancing:"


	assert(sorted(label_count_dict.keys()) == sorted(label_features_dict.keys()))

	for label in label_count_dict.keys():
		label_mean_feat_dict[label] = np.mean(label_features_dict[label], axis=0)

	minors = []
	majors = []
	mean_count = math.floor(np.mean(np.array([label_count_dict[k] for k in label_count_dict.keys()])))
	for label in label_count_dict.keys():
		if label_count_dict[label] < mean_count:
			minors.append(label)
		else:
			majors.append(label)

	############################################################
	im_selection = []
	for cnt in range(al_budget):
		im_candidates = []
		for i, feature in enumerate(candidate_features):
			if i not in im_selection:
				try:
					d_min = min([np.linalg.norm(feature - label_mean_feat_dict[min_label]) for min_label in minors])
					d_maj = min([np.linalg.norm(feature - label_mean_feat_dict[max_label]) for max_label in majors])

					# diff = [d1 - d2 for d1, d2 in zip(d_min, d_max)]
					diff = d_min - d_maj
					if diff <= 0:
						im_candidates.append(i)
				except:
					im_candidates.append(i)

			if len(im_candidates) == 0 : #all candidates are nearer to major classes
				im_candidates = list(set(range(len(candidate_features))) - set(im_selection))
				assert(len(list(set(im_candidates) & set(im_selection))) == 0 )

		if cnt == 0 :
			best_idx = im_candidates[random.randint(0, len(im_candidates) - 1)]
			im_selection.append(best_idx)

		elif cnt == 1 :
			best_idx = im_candidates[
				np.argmax([dist_matrix[im_selection[0], idx] for idx in im_candidates])
			]
			im_selection.append(best_idx)
		else:
			best_idx = im_candidates[
				np.argmax(
					dist_matrix[np.ix_(im_selection, im_candidates)].min(axis=0)
				)
			]
			im_selection.append(best_idx)

		#annoatte the image
		true_label = candidate_labels[best_idx]
		true_feature = candidate_features[best_idx]

		if not true_label in label_count_dict.keys():
			label_count_dict[true_label] = 1
			label_features_dict[true_label] = true_feature
		else:
			label_count_dict[true_label] += 1
			label_features_dict[true_label] = np.vstack((label_features_dict[true_label], true_feature))

		label_mean_feat_dict[true_label] = np.mean(label_features_dict[true_label], axis=0)

		minors = []
		majors = []
		mean_count = math.floor(np.mean(np.array([label_count_dict[k] for k in label_count_dict.keys()])))
		for label in label_count_dict.keys():
			if label_count_dict[label] < mean_count:
				minors.append(label)
			else:
				majors.append(label)

	features = candidate_features[im_selection]
	labels = candidate_labels[im_selection]
	paths = candidate_paths[im_selection]

	return labels, paths, features
