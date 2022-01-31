#!/usr/bin/env python
# coding: utf-8
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from Bio.PDB import *
# import spektral
import dgl
import dgl.nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy
import shutil

#import bio_embeddings.embed # yannick
# from bio_embeddings.embed import OneHotEncodingEmbedder # yannick
#from code.File_extractor import WD

import warnings

warnings.filterwarnings("ignore")

# supplements from previous script
#train_graphs = os.path.join("../ML_data", "train_data", "graphs")
#val_graphs = os.path.join("../ML_data", "val_data", "graphs")
emb_name = "glove"
#train_embs = os.path.join("../ML_data", "train_data", "embeddings", emb_name)
#val_embs = os.path.join("../ML_data", "val_data", "embeddings", emb_name)


# Load the graphs
from pathlib import Path
import pickle

TRAIN_GRAPHS = {}
VAL_GRAPHS = {}
TEST_GRAPHS = {}
WD = Path(__file__).resolve().parents[1]
train_graphs = WD / "ML_data" / "train_data" / "graphs"
val_graphs = WD / "ML_data" / "val_data" / "graphs"
test_graphs = WD / "ML_data" / "test_data" / "graphs"

train_embs = WD / "ML_data" / "train_data" / "embeddings" / emb_name
val_embs = WD / "ML_data" / "val_data" / "embeddings" / emb_name
test_embs = WD / "ML_data" / "test_data" / "embeddings" / emb_name
for file in os.listdir(train_graphs):
    TRAIN_GRAPHS[file[:-2]] = pickle.load(open(os.path.join(train_graphs, file), "rb"))
print(f"Training graphs loaded : {len(TRAIN_GRAPHS)} items")
for file in os.listdir(val_graphs):
    VAL_GRAPHS[file[:-2]] = pickle.load(open(os.path.join(val_graphs, file), "rb"))
print(f"Validation graphs loaded : {len(VAL_GRAPHS)} items")
for file in os.listdir(test_graphs):
    TEST_GRAPHS[file[:-2]] = pickle.load(open(os.path.join(test_graphs, file), "rb"))
print(f"Testing graphs loaded : {len(TEST_GRAPHS)} items")

# "In[26]:"


for graph in TRAIN_GRAPHS.values():
    A, labels = graph
    break

# "In[29]:"


labels.shape

# "In[34]:"


# Load the embeddings

import pickle

TRAIN_EMBS = {}
VAL_EMBS = {}
TEST_EMBS = {}
for file in os.listdir(train_embs):
    TRAIN_EMBS[file[:-2]] = pickle.load(open(os.path.join(train_embs, file), "rb"))

print(f"Training embeddings loaded : {len(TRAIN_EMBS)} items")
for file in os.listdir(val_embs):
    VAL_EMBS[file[:-2]] = pickle.load(open(os.path.join(val_embs, file), "rb"))

print(f"Validation embeddings loaded : {len(VAL_EMBS)} items")

for file in os.listdir(test_embs):
    TEST_EMBS[file[:-2]] = pickle.load(open(os.path.join(test_embs, file), "rb"))

print(f"Testing embeddings loaded : {len(TEST_EMBS)} items")

# "In[228]:"


for em in TRAIN_EMBS.values():
    print(em.shape)
    break


# "In[35]:"
import tensorflow as tf

def create_dgl_graph(A, feats, labels, train, val, test):
    g = dgl.from_scipy(A)
    feats = feats.reset_index(drop=True)
    g.ndata["feat"] = torch.tensor(feats[feats.columns].values).long()
    g.ndata["label"] = torch.tensor(labels).long()
    g.ndata["train_mask"] = torch.tensor(np.ones(A.shape[0]) == train)
    g.ndata["val_mask"] = torch.tensor(np.ones(A.shape[0]) == val)
    g.ndata["test_mask"] = torch.tensor(np.ones(A.shape[0]) == test)

    return g


def create_dgl_data(graphs, embeddings, train, val, test, onehot, mode="sum"):
    g = dgl.graph([])
    mode = "glove"
    for prot in graphs.keys():
        A, labels = graphs[prot]
        feats = embeddings[prot]
        if mode == "sum":
            print("mode sum")
            feats = np.sum(embeddings[prot], axis=0)
        elif mode == "concat":
            print("mode concat")
            feats = embeddings[prot].reshape(embeddings[prot].shape[1], -1)
        elif onehot: # yannick
            print("onehot not using")
            feats = feats[mode]

        g1 = create_dgl_graph(A, feats, labels, train, val, test)
        g = dgl.batch([g, g1])

    return g


# "In[39]:"


n_res = 255
A = np.zeros((n_res, n_res))
labels = np.zeros(n_res)
feats = np.random.rand(n_res, 21)

A[1, 3] = 1
A[17, 32] = 1
A[136, 32] = 1
A[0, 85] = 1
A[99, 17] = 1

labels[[3, 21, 18, 22]] = 1


A = scipy.sparse.csr_matrix(A)
#A, labels, feats

# "In[48]:"


import torch

#my_graph = create_dgl_graph(A, feats, labels, 1, 0, 0)
#print(my_graph.ndata)
# "In[244]:"
# Split into validation test

# GRAPHS
i = 0
VALIDATION_GRAPHS = {}
TEST_GRAPHS = {}
for x, y in VAL_GRAPHS.items():
    if i < len(VAL_GRAPHS):
        VALIDATION_GRAPHS[x] = y
    else:
        TEST_GRAPHS[x] = y
    i += 1

# EMBEDDINGS
i = 0
VALIDATION_EMBS = {}
TEST_EMBS = {}
for x, y in VAL_GRAPHS.items():
    if i < len(VAL_GRAPHS):
        VALIDATION_GRAPHS[x] = y
    else:
        TEST_GRAPHS[x] = y
    i += 1


# "In[245]:"

create = False

train_embs_pth = os.path.join("../ML_data", "train_data", "graph")
val_embs_pth = os.path.join("../ML_data", "val_data", "graph")
test_embs_pth = os.path.join("../ML_data", "test_data", "graph")
if create:
    # Train
    onehot = False
    print("Creating training set...")
    g_train = create_dgl_data(TRAIN_GRAPHS, TRAIN_EMBS, train=1, val=0, test=0, onehot=onehot)
    pickle.dump(g_train, open(os.path.join(train_embs_pth), "wb"))
    # Validation
    print("Creating validation set...")
    g_val = create_dgl_data(VAL_GRAPHS, VAL_EMBS, train=0, val=1, test=0, onehot=onehot)
    pickle.dump(g_val, open(os.path.join(val_embs_pth), "wb"))
    # Test
    print("Creating Test set...")
    g_test = create_dgl_data(TEST_GRAPHS, TEST_EMBS, train=0, val=0, test=1, onehot=onehot)
    pickle.dump(g_test, open(os.path.join(test_embs_pth), "wb"))
else:
    g_train = pickle.load(open(train_embs_pth, "rb"))
    g_val = pickle.load(open(val_embs_pth, "rb"))
    g_test = pickle.load(open(test_embs_pth, "rb"))
g = dgl.batch([g_train, g_val, g_test])

# "In[246]:"


labels = g.ndata['label']
train_mask = g.ndata['train_mask']
val_mask = g.ndata['val_mask']
test_mask = g.ndata['test_mask']
print("Distribution of training labels (all proteins)")
print(np.unique(np.array(labels[train_mask]), return_counts=True))
print("Distribution of validation labels (all proteins)")
print(np.unique(np.array(labels[val_mask]), return_counts=True))
print("Distribution of testing labels (all proteins)")
print(np.unique(np.array(labels[test_mask]), return_counts=True))
# "In[253]:"


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


# "In[324]:"


# Graph convolution layer
from dgl.nn import SGConv as ConvLayer
from dgl.nn import MaxPooling


class GCN(nn.Module):

    def __init__(self, layers):
        super(GCN, self).__init__()
        self.convs = []
        self.n_layers = len(layers) - 1
        # Hidden layers
        self.conv1 = ConvLayer(layers[0], layers[1])  # ,norm='both',allow_zero_in_degree=True)
        if self.n_layers >= 2:
            self.conv2 = ConvLayer(layers[1], layers[2])  # ,norm='both')#,allow_zero_in_degree=True)
        if self.n_layers >= 3:
            self.conv3 = ConvLayer(layers[2], layers[3])  # ,norm='both')#,allow_zero_in_degree=True)
        if self.n_layers >= 4:
            self.conv4 = ConvLayer(layers[3], layers[4])  # ,norm='both')#,allow_zero_in_degree=True)
        if self.n_layers >= 5:
            self.conv5 = ConvLayer(layers[4], layers[5])  # ,norm='both')#,allow_zero_in_degree=True)
        if self.n_layers >= 6:
            self.conv6 = ConvLayer(layers[5], layers[6])  # ,norm='both')#,allow_zero_in_degree=True)
        if self.n_layers >= 7:
            self.conv7 = ConvLayer(layers[6], layers[7])  # ,norm='both')#,allow_zero_in_degree=True)
        if self.n_layers >= 8:
            self.conv8 = ConvLayer(layers[7], layers[8])  # ,norm='both')#,allow_zero_in_degree=True)
        if self.n_layers >= 9:
            self.conv9 = ConvLayer(layers[8], layers[9])  # ,norm='both')#,allow_zero_in_degree=True)

        # Output layer
        self.output = ConvLayer(layers[-1], 3)  # ,norm='both')#,allow_zero_in_degree=True)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        n = layers[1]
        if self.n_layers >= 2:
            h = self.conv2(g, h)
            h = F.relu(h)
            n = layers[2]
        if self.n_layers >= 3:
            h = self.conv3(g, h)
            h = F.relu(h)
            n = layers[3]
        if self.n_layers >= 4:
            h = self.conv4(g, h)
            h = F.relu(h)
            n = layers[4]
        if self.n_layers >= 5:
            h = self.conv5(g, h)
            h = F.relu(h)
            n = layers[5]
        h = nn.BatchNorm1d(n)(h)
        h = self.output(g, h)
        h = F.softmax(h, 1)
        return h

def slim_for_metrics(features, pred, labels, train_mask, val_mask):
    df_feat = pd.DataFrame(features.numpy())
    df_pred = pd.DataFrame(pred.numpy())
    df_labels = pd.DataFrame(labels.numpy())
    df_train_mask = pd.DataFrame(train_mask.numpy())
    df_val_mask = pd.DataFrame(val_mask.numpy())
    usable = df_feat[512] == 1
    df_pred = df_pred[usable]
    df_labels = df_labels[usable]
    df_train_mask = df_train_mask[usable]
    df_val_mask = df_val_mask[usable]
    met_pred = torch.tensor(df_pred.to_numpy())
    met_labels = torch.tensor(df_labels.to_numpy())
    met_train_mask = torch.tensor(df_train_mask.to_numpy())
    met_val_mask = torch.tensor(df_val_mask.to_numpy())
    return met_pred, met_labels, met_train_mask, met_val_mask

#"In[325]:"


def train(g, model, n_epochs, metric_name, lr=3e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr)
    # Get graph features, labels and masks
    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']

    #     # Set sample importance weights
    weights = []
    # n0 = sum(x == 0 for x in labels[train_mask])
    n1 = sum(x == 1 for x in labels[train_mask])
    n2 = sum(x == 2 for x in labels[train_mask])
    for e in range(n_epochs + 1):
        print("Epoch: ", e)
        # Forward
        logits = model(g, features)
        # Compute prediction
        pred = logits.argmax(1)
        # Compute loss
        # Note that you should only compute the losses of the nodes in the training set.
        weight = (torch.Tensor([0, n2, n1]))
        loss = nn.CrossEntropyLoss(weight)(logits[train_mask].float(), labels[train_mask].reshape(-1, ).long())

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if e % 5 == 0:
            # removing the impossible aa from the prediction
            met_pred, met_labels, met_train_mask, met_val_mask = slim_for_metrics(features, pred, labels, train_mask, val_mask)
            # Evaluation metric
            train_metric, val_metric = get_metric(met_pred, met_labels, met_train_mask, met_val_mask, metric_name)
            print('In epoch {}, loss: {:.3f}, train {} : {:.3f} , val {} : {:.3f}'.format(
                e, loss, metric_name, train_metric, metric_name, val_metric))
            break

    return np.array(pred[val_mask]), np.array(labels[val_mask])


# ## Baseline (onehot encoding)

# #### Chen Dataset (200 train , 51 validation)

# "In[323]:"


# Train the model
layers = [g.ndata['feat'].shape[1],64,16,8]#,64,32,16,8,4] # , 64] # yannick
print("model : ", layers)
model = GCN(layers)
import cProfile
cProfile.run("pred, true = train(g, model, n_epochs=500, metric_name='mcc')", "output.txt")
pred, true = train(g, model, n_epochs=500, metric_name="mcc")

# #### Holo4k

# ## Seqvec (sum)

# #### Chen Dataset (200 train , 51 validation)

# "In[317]:"


# Train the model
layers = [g.ndata['feat'].shape[1], 2048]
print("model : ", layers)
model = GCN(layers)
pred, true = train(g, model, n_epochs=500, metric_name="mcc")

# "In[ ]:"




