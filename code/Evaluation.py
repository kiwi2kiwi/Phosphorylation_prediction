#!/usr/bin/env python
# coding: utf-8
import pickle
import os
from pathlib import Path
import numpy as np
np.random.seed(1)
import pandas as pd
import dgl
dgl.seed(1)
import dgl.nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy

import GNN_architect


import warnings

warnings.filterwarnings("ignore")

train_embs_pth = os.path.join("../ML_data", "train_data", "graph")
val_embs_pth = os.path.join("../ML_data", "val_data", "graph")
test_embs_pth = os.path.join("../ML_data", "test_data", "graph")
cv_supplements_pth = os.path.join("../ML_data", "cv_supplements")
name_dict = os.path.join("../ML_data", "name_dict")
g_train = pickle.load(open(train_embs_pth, "rb"))
g_val = pickle.load(open(val_embs_pth, "rb"))
g_test = pickle.load(open(test_embs_pth, "rb"))
cv_s = pickle.load(open(cv_supplements_pth, "rb"))
name_position_dict = pickle.load(open(os.path.join(name_dict), "rb"))
graph_lengths, graph_number, test_graph_feats = cv_s[0], cv_s[1], cv_s[2]

g = dgl.batch([g_train, g_val, g_test])
del g_train
del g_val
del g_test


from sklearn.metrics import recall_score, accuracy_score, precision_score
from sklearn.metrics import matthews_corrcoef as MCC


def get_metric(pred, labels, test_mask, name):
    # Accuracy
    if name == "accuracy":
        train_acc = accuracy_score(np.array(labels[test_mask]), np.array(pred[test_mask]))
        return train_acc

    # Recall
    if name == "recall":
        train_rec = recall_score(np.array(labels[test_mask]), np.array(pred[test_mask]), average='macro')
        return train_rec

    # Precision
    if name == "precision":
        train_prec = precision_score(np.array(labels[test_mask]), np.array(pred[test_mask]), average='macro')
        return train_prec
    # Matthews Correlation Coefficient (MCC)
    if name == "mcc":
        train_mcc = MCC(np.array(labels[test_mask]), np.array(pred[test_mask]))
        return train_mcc

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# Get graph features, labels and masks
features = g.ndata['feat']
labels = g.ndata['label']
train_mask = g.ndata['train_mask']
val_mask = g.ndata['val_mask']
test_mask = g.ndata['test_mask']

model = pickle.load(open(os.path.join("..\\ML_data\\ML_models_saves\\model,8,4.pt"), "rb"))
# evaluation mode
model.eval()
# Forward
logits = model(g, features)
# Compute prediction
pred = logits.argmax(1)

pP_eval = pd.DataFrame(columns=["name", "train_metric", "val_metric"])
df_feat = pd.DataFrame(features.numpy())
usable = df_feat[512] == 1
pred_df = pd.DataFrame(pred)
labels_df = pd.DataFrame(labels.numpy())
test_mask_df = pd.DataFrame(test_mask)
pred_df = pred_df[usable]
labels_df = labels_df[usable]
test_mask_df = test_mask_df[usable]

testing_metric_list = []

for key in name_position_dict.keys():
    value = name_position_dict[key]
    start = value[0]
    end = value[1]
    name = value[2]
    if value[3] == 2:  # testing
        temp_pred_df = pred_df.loc[start:end, :]
        temp_labels_df = labels_df.loc[start:end, :]
        temp_test_mask_df = test_mask_df.loc[start:end, :]
        pred_tp = torch.tensor(temp_pred_df.to_numpy())
        labels_tp = torch.tensor(temp_labels_df.to_numpy())
        test_mask_tp = torch.tensor(temp_test_mask_df.to_numpy())
        # print("metric on: " + name)
        # print("predictions:" + str(met_pred[start:end]))
        # print("labels:" + str(met_labels[start:end]))
        test_metric = get_metric(pred_tp, labels_tp, test_mask_tp, "mcc")
        testing_metric_list.append(test_metric)
print("test")