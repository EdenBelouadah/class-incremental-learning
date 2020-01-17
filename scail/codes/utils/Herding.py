import os
from numpy.linalg import norm
import numpy as np
import time, math

class StaticHerding():
    def __init__(self):
        pass

    def compute_rebuffi_herding_faster(self, exemplars_dir, img_list_path, img_feats_path, per_class, out_file_name):
        if not os.path.isdir(exemplars_dir):
            os.makedirs(exemplars_dir)

        rebuffi_exem = os.path.join(exemplars_dir, out_file_name)
        f_rebuffi = open(rebuffi_exem, "w")
        # open the list of images and put the labels and associated paths in a list
        pos_list = []
        line_cnt = 0
        f_list = open(img_list_path)
        for line in f_list:
            pos_list.append(line.rstrip())
        f_list.close()
        # open the features file and store the features and the image paths in dictionaries with class labels as keys
        feat_dict = {}
        path_dict = {}
        ordered_labels = []  # list to store the labels in the original order from the initial list
        feat_cnt = 0
        f_feats = open(img_feats_path)
        for feat_line in f_feats:
            if 1 == 1:  # TO DO put all features back here
                # print feat_cnt
                feat_line = feat_line.rstrip()
                # get the path and the label
                crt_parts = pos_list[feat_cnt].split(" ")
                label = crt_parts[1]
                path = crt_parts[0]
                if not label in feat_dict:
                    feat_dict[label] = feat_line
                    path_dict[label] = path
                    ordered_labels.append(label)
                else:
                    feat_dict[label] = feat_dict[label] + "\t" + feat_line
                    path_dict[label] = path_dict[label] + "\t" + path
            feat_cnt = feat_cnt + 1
        f_feats.close()
        for ll in range(0, len(ordered_labels)):  # TO DO run for all labels here
            label = ordered_labels[ll]
            # get the features and images associated to the current label
            feat_list = feat_dict[label].split("\t")
            img_list = path_dict[label].split("\t")
            # print "rebuffi started class:",ll,", class images:",len(img_list)
            # compute the mean representation of the class. normalize L2 each feature before mean computation
            sum_feat = []  # list for the sum of feature dimensions
            feat_size = feat_list[0].split(" ")
            # initialize the sum list
            for fs in range(0, len(feat_size)):
                sum_feat.append(0)
            feat_list_norm = []
            for feat in feat_list:
                np_feat = np.fromstring(feat, dtype=float, sep=' ')
                l2_norm = norm(np_feat)
                crt_feat_norm = ""
                local_feat = []
                for dd in range(0, np_feat.shape[0]):
                    norm_dim = np_feat[dd] / l2_norm
                    # crt_feat_norm = crt_feat_norm+" "+str(norm_dim)
                    local_feat.append(norm_dim)
                    sum_feat[dd] = sum_feat[dd] + norm_dim
                # crt_feat_norm.append(local_feat)
                feat_list_norm.append(local_feat)
            # compute the mean feature for the class
            mean_interm = []
            mean_norm = 0
            for dd in range(0, len(sum_feat)):
                dim_mean = sum_feat[dd] / len(feat_list)
                mean_interm.append(dim_mean)
                mean_norm = mean_norm + dim_mean * dim_mean
            mean_norm = math.sqrt(mean_norm)
            mean_final = []  # list to store the final form of the mean which is L2-normalized
            for dd in range(0, len(mean_interm)):
                l2_dim_mean = mean_interm[dd] / mean_norm
                mean_final.append(l2_dim_mean)
            # print mean_interm,"\n",mean_final
            # sort the features by their aggregate distance to the true mean of the class.
            # in each step, all remaining features are tested and the one that minimized the current distance of means is kept
            ranked_exemplars = {}  # dict for exemplars ranked by their distance to the mean of the class
            aggregate_sum = []  # list to store the aggregate mean of exemplars that were already chosen
            for fs in range(0, len(feat_size)):
                aggregate_sum.append(0)
            time_start = time.time()
            # cumul_diff_dist= 0
            # cumul_diff_prep = 0
            # cumul_finalize = 0
            # cumul_diff_min = 0
            # cumul_compute = 0
            # print "started reranking"
            # next two lines to ensure that the number of exemplars is never bigger than the total number of images per class
            if per_class > len(feat_list_norm):
                per_class = len(feat_list_norm)
            while (len(ranked_exemplars) < per_class):  # TO DO put the real number of exemplars here
                # print len(ranked_exemplars),per_class
                feat_cnt = 0
                min_dist = 10000000.0  # large initial value for the min distance
                crt_candidate = ""  # path to the current image candidate
                feat_candidate = []  # tmp list to store the features of the current candidate
                for crt_feat in feat_list_norm:
                    crt_img = img_list[feat_cnt]
                    # check if the current image is in the dictionary of selected exemplars
                    if not crt_img in ranked_exemplars:
                        # time_start_compute = time.time()
                        l2_dist = 0
                        norm_crt_feats = []
                        count_zeros = 0
                        # time_prep_start = time.time()
                        tmp_norm = 0
                        tmp_interm = []
                        crt_exemplars = float(len(ranked_exemplars) + 1)
                        for dd in range(0, len(crt_feat)):
                            dim_sum = aggregate_sum[dd] + crt_feat[dd]
                            tmp_interm.append(dim_sum)
                            tmp_norm = tmp_norm + dim_sum * dim_sum
                        tmp_norm = math.sqrt(tmp_norm / crt_exemplars)
                        # time_prep_stop = time.time()
                        # time_finalize_start = time.time()
                        tmp_final = []
                        # compute the distance between the current aggregate mean and the mean of the class
                        feat_dist = 0
                        for dd in range(0, len(tmp_interm)):
                            l2_dim_mean = tmp_interm[dd] / tmp_norm
                            diff_dim = l2_dim_mean - mean_final[dd]
                            feat_dist = feat_dist + diff_dim * diff_dim
                            # tmp_final.append(l2_dim_mean)
                        # time_finalize_stop = time.time()
                        # cumul_finalize = cumul_finalize + time_finalize_stop - time_finalize_start
                        # cumul_diff_prep = cumul_diff_prep + time_prep_stop - time_prep_start
                        # time_start_min = time.time()
                        if feat_dist < min_dist:
                            min_dist = feat_dist
                            crt_candidate = crt_img
                            # print min_dist,crt_candidate
                            feat_candidate = []  # reinitialize the feats for the new candidate
                            # update the feats for the candidate
                            for nd in crt_feat:
                                feat_candidate.append(nd)
                        # time_stop_min = time.time()
                        # cumul_diff_min = cumul_diff_min + time_stop_min - time_start_min
                        # time_stop_compute = time.time()
                        # cumul_compute = cumul_compute + time_stop_compute - time_start_compute
                    feat_cnt = feat_cnt + 1
                # update the dictionary of exemplars
                ranked_exemplars[crt_candidate] = len(ranked_exemplars)
                # update the aggregate sum list with the features of the current candidate
                for nd in range(0, len(feat_candidate)):
                    aggregate_sum[nd] = aggregate_sum[nd] + feat_candidate[nd]
            for img_path, dist in sorted(ranked_exemplars.iteritems(), key=lambda (k, v): (v, k), reverse=False):
                to_out = img_path + " " + ordered_labels[ll]
                f_rebuffi.write(to_out + "\n")
            time_stop = time.time()
            time_diff = time_stop - time_start
            # print "execution time:",time_diff,", preparation:",cumul_diff_prep,", dist computation:",cumul_finalize,", total compute time:",cumul_compute
            # print "execution time:",time_dif

        f_rebuffi.close()

    def load_class_paths(self, class_number, batch_paths):
        batch_paths = open(batch_paths, 'r').readlines()
        class_paths = []
        for path_line in batch_paths:
            image_path = path_line.strip()
            path_line = path_line.strip().split()
            image_class = int(path_line[1])
            if image_class == class_number:
                class_paths.append(image_path)
        return class_paths

    def reduce_exemplars(self, exemplars_dir, old_classes_number, per_class, batch_number, file_names_suffix):
        previous_exemplars_file = os.path.join(exemplars_dir, str(batch_number)+'_old'+file_names_suffix)
        current_exemplars_file = os.path.join(exemplars_dir, str(batch_number+1)+'_old'+file_names_suffix)
        current_exemplars = open(current_exemplars_file, 'a')
        for label in range(old_classes_number):  # old class
            class_paths = self.load_class_paths(label, previous_exemplars_file)
            for img_path in class_paths[:per_class]:
                current_exemplars.write(img_path + "\n")
        current_exemplars.close()















# def __iter__(self):
#     print ('\tcalling Sampler:__iter__')
#     return iter(range(self.num_samples))
#
# def __len__(self):
#     print ('\tcalling Sampler:__len__')
#     return self.num_samples
