#!/usr/bin/env python
# coding=utf-8
import torch
import numpy as np
from utils_pytorch import *

def compute_features(tg_feature_model, evalloader, num_samples, num_features, device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tg_feature_model.eval()

    features = np.zeros([num_samples, num_features])
    start_idx = 0
    with torch.no_grad():
        for inputs, targets in evalloader:
            inputs = inputs.to(device)
            features[start_idx:start_idx+inputs.shape[0], :] = np.squeeze(tg_feature_model(inputs).cpu())
            start_idx = start_idx+inputs.shape[0]
    assert(start_idx==num_samples)
    return features
