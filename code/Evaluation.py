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

import warnings

warnings.filterwarnings("ignore")

train_embs_pth = os.path.join("../ML_data", "train_data", "graph")
val_embs_pth = os.path.join("../ML_data", "val_data", "graph")
test_embs_pth = os.path.join("../ML_data", "test_data", "graph")
cv_supplements_pth = os.path.join("../ML_data", "cv_supplements")
g_train = pickle.load(open(train_embs_pth, "rb"))
g_val = pickle.load(open(val_embs_pth, "rb"))
g_test = pickle.load(open(test_embs_pth, "rb"))
cv_s = pickle.load(open(cv_supplements_pth, "rb"))
graph_lengths, graph_number, test_graph_feats = cv_s[0], cv_s[1], cv_s[2]

g = dgl.batch([g_train, g_val, g_test])
del g_train
del g_val
del g_test


from sklearn.metrics import recall_score, accuracy_score, precision_score
from sklearn.metrics import matthews_corrcoef as MCC


def get_metric(pred, labels, train_mask, val_mask, name):
    # Accuracy
    if name == "accuracy":
        train_acc = accuracy_score(np.array(labels[train_mask]), np.array(pred[train_mask]))
        val_acc = accuracy_score(np.array(labels[val_mask]), np.array(pred[val_mask]))
        return train_acc, val_acc

    # Recall
    if name == "recall":
        train_rec = recall_score(np.array(labels[train_mask]), np.array(pred[train_mask]), average='macro')
        val_rec = recall_score(np.array(labels[val_mask]), np.array(pred[val_mask]), average='macro')
        return train_rec, val_rec

    # Precision
    if name == "precision":
        train_prec = precision_score(np.array(labels[train_mask]), np.array(pred[train_mask]), average='macro')
        val_prec = precision_score(np.array(labels[val_mask]), np.array(pred[val_mask]), average='macro')
        return train_prec, val_prec
    # Matthews Correlation Coefficient (MCC)
    if name == "mcc":
        train_mcc = MCC(np.array(labels[train_mask]), np.array(pred[train_mask]))
        val_mcc = MCC(np.array(labels[val_mask]), np.array(pred[val_mask]))
        return train_mcc, val_mcc

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

model.eval()
val_loss = nn.CrossEntropyLoss(weight)(logits[val_mask].float(), labels[val_mask].reshape(-1, ).long())

optimizer.zero_grad()
loss.backward()
optimizer.step()

pP_eval = pd.DataFrame(columns=["name", "train_metric", "val_metric"])
df_feat = pd.DataFrame(features.numpy())
usable = df_feat[512] == 1
pred_df = pd.DataFrame(pred)
labels_df = pd.DataFrame(labels.numpy())
train_mask_df = pd.DataFrame(train_mask)
val_mask_df = pd.DataFrame(val_mask)
pred_df = pred_df[usable]
labels_df = labels_df[usable]
train_mask_df = train_mask_df[usable]
val_mask_df = val_mask_df[usable]
for key in name_position_dict.keys():
    value = name_position_dict[key]
    start = value[0]
    end = value[1]
    name = value[2]

    temp_pred_df = pred_df.loc[start:end, :]
    temp_labels_df = labels_df.loc[start:end, :]
    temp_train_mask_df = train_mask_df.loc[start:end, :]
    temp_val_mask_df = val_mask_df.loc[start:end, :]
    pred_tp = torch.tensor(temp_pred_df.to_numpy())
    labels_tp = torch.tensor(temp_labels_df.to_numpy())
    train_mask_tp = torch.tensor(temp_train_mask_df.to_numpy())
    val_mask_tp = torch.tensor(temp_val_mask_df.to_numpy())
    # print("metric on: " + name)
    # print("predictions:" + str(met_pred[start:end]))
    # print("labels:" + str(met_labels[start:end]))
    train_metric, val_metric = get_metric(pred_tp, labels_tp, train_mask_tp, val_mask_tp, "mcc")
