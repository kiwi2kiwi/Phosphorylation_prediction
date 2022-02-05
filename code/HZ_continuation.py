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
CV_GRAPHS = {}
WD = Path(__file__).resolve().parents[1]
train_graphs = WD / "ML_data" / "train_data" / "graphs"
val_graphs = WD / "ML_data" / "val_data" / "graphs"
test_graphs = WD / "ML_data" / "test_data" / "graphs"
cv_graphs = WD / "ML_data" / "cv_data" / "graphs"

train_embs = WD / "ML_data" / "train_data" / "embeddings" / emb_name
val_embs = WD / "ML_data" / "val_data" / "embeddings" / emb_name
test_embs = WD / "ML_data" / "test_data" / "embeddings" / emb_name
cv_embs = WD / "ML_data" / "cv_data" / "embeddings" / emb_name
for file in os.listdir(train_graphs):
    TRAIN_GRAPHS[file[:-2]] = pickle.load(open(os.path.join(train_graphs, file), "rb"))
print(f"Training graphs loaded : {len(TRAIN_GRAPHS)} items")
for file in os.listdir(val_graphs):
    VAL_GRAPHS[file[:-2]] = pickle.load(open(os.path.join(val_graphs, file), "rb"))
print(f"Validation graphs loaded : {len(VAL_GRAPHS)} items")
for file in os.listdir(test_graphs):
    TEST_GRAPHS[file[:-2]] = pickle.load(open(os.path.join(test_graphs, file), "rb"))
print(f"Testing graphs loaded : {len(TEST_GRAPHS)} items")
for file in os.listdir(cv_graphs):
    CV_GRAPHS[file[:-2]] = pickle.load(open(os.path.join(cv_graphs, file), "rb"))
print(f"Training graphs loaded : {len(CV_GRAPHS)} items")

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
CV_EMBS = {}
for file in os.listdir(train_embs):
    TRAIN_EMBS[file[:-2]] = pickle.load(open(os.path.join(train_embs, file), "rb"))

print(f"Training embeddings loaded : {len(TRAIN_EMBS)} items")
for file in os.listdir(val_embs):
    VAL_EMBS[file[:-2]] = pickle.load(open(os.path.join(val_embs, file), "rb"))

print(f"Validation embeddings loaded : {len(VAL_EMBS)} items")

for file in os.listdir(test_embs):
    TEST_EMBS[file[:-2]] = pickle.load(open(os.path.join(test_embs, file), "rb"))

print(f"Testing embeddings loaded : {len(TEST_EMBS)} items")

for file in os.listdir(cv_embs):
    CV_EMBS[file[:-2]] = pickle.load(open(os.path.join(cv_embs, file), "rb"))

print(f"CV embeddings loaded : {len(CV_EMBS)} items")

# "In[228]:"


for em in CV_EMBS.values():
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

def create_dgl_graph_cv(A, feats, labels, splits, test):
    g = dgl.from_scipy(A)
    feats = feats.reset_index(drop=True)
    g.ndata["feat"] = torch.tensor(feats[feats.columns].values).long()
    g.ndata["label"] = torch.tensor(labels).long()
    g.ndata["cv_train_mask"] = torch.tensor(np.ones(A.shape[0]) == splits[0])
    g.ndata["cv_val"] = torch.tensor(np.ones(A.shape[0]) == splits[0])
    # g.ndata["split_2_mask"] = torch.tensor(np.ones(A.shape[0]) == splits[1])
    # g.ndata["split_3_mask"] = torch.tensor(np.ones(A.shape[0]) == splits[2])
    # g.ndata["split_4_mask"] = torch.tensor(np.ones(A.shape[0]) == splits[3])
    # g.ndata["split_5_mask"] = torch.tensor(np.ones(A.shape[0]) == splits[4])
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

def create_dgl_data_cv(graphs, embeddings, cv_splits, test, onehot, mode="sum"):
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

        g1 = create_dgl_graph_cv(A, feats, labels, [0,1,2,3,4], test)
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

# # GRAPHS
# i = 0
# VALIDATION_GRAPHS = {}
# TEST_GRAPHS = {}
# for x, y in VAL_GRAPHS.items():
#     if i < len(VAL_GRAPHS):
#         VALIDATION_GRAPHS[x] = y
#     else:
#         TEST_GRAPHS[x] = y
#     i += 1
#
# # EMBEDDINGS
# i = 0
# VALIDATION_EMBS = {}
# TEST_EMBS = {}
# for x, y in VAL_GRAPHS.items():
#     if i < len(VAL_GRAPHS):
#         VALIDATION_GRAPHS[x] = y
#     else:
#         TEST_GRAPHS[x] = y
#     i += 1




# "In[245]:"

create = False

train_embs_pth = os.path.join("../ML_data", "train_data", "graph")
val_embs_pth = os.path.join("../ML_data", "val_data", "graph")
test_embs_pth = os.path.join("../ML_data", "test_data", "graph")
cv_embs_pth = os.path.join("../ML_data", "cv_data", "graph")
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
    # cv
    print("Creating cv set...")
    cv_splits = [0,1,2,3,4]
    g_cv = create_dgl_data_cv(CV_GRAPHS, CV_EMBS, cv_splits, test=5, onehot=onehot)
    pickle.dump(g_cv, open(os.path.join(cv_embs_pth), "wb"))
else:
    g_train = pickle.load(open(train_embs_pth, "rb"))
    g_val = pickle.load(open(val_embs_pth, "rb"))
    g_test = pickle.load(open(test_embs_pth, "rb"))
    g_cv = pickle.load(open(cv_embs_pth, "rb"))

cv = True
if not cv:
    g = dgl.batch([g_train, g_val, g_test])
if cv:
    g = dgl.batch([g_cv, g_test])


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
dgl.seed(1)

layers = []
class GCN(nn.Module):

    def __init__(self, layers):
        super(GCN, self).__init__()
        self.convs = []
        self.n_layers = len(layers) - 1
        self.layers = layers
        # Hidden layers
        self.conv1 = ConvLayer(layers[0], layers[1], allow_zero_in_degree=True)#, bias=False)#, k=2) #  , norm='both',
        if self.n_layers >= 2:
            self.conv2 = ConvLayer(layers[1], layers[2], allow_zero_in_degree=True)#, bias=False)#, k=2) #  , norm='both',
        if self.n_layers >= 3:
            self.conv3 = ConvLayer(layers[2], layers[3], allow_zero_in_degree=True)#, bias=False)#, k=2) #  , norm='both',
        if self.n_layers >= 4:
            self.conv4 = ConvLayer(layers[3], layers[4], allow_zero_in_degree=True)#, bias=False)#, k=2) #  , norm='both',
        if self.n_layers >= 5:
            self.conv5 = ConvLayer(layers[4], layers[5], allow_zero_in_degree=True)#, bias=False)#, k=2) #  , norm='both',
        if self.n_layers >= 6:
            self.conv6 = ConvLayer(layers[5], layers[6], allow_zero_in_degree=True)#, bias=False)#, k=2) #  , norm='both',
        if self.n_layers >= 7:
            self.conv7 = ConvLayer(layers[6], layers[7], allow_zero_in_degree=True)#, bias=False)#, k=2) #  , norm='both',
        if self.n_layers >= 8:
            self.conv8 = ConvLayer(layers[7], layers[8], allow_zero_in_degree=True)#, bias=False)#, k=2) #  , norm='both',
        if self.n_layers >= 9:
            self.conv9 = ConvLayer(layers[8], layers[9], allow_zero_in_degree=True)#, bias=False)#, k=2) #  , norm='both',

        # Output layer
        self.output = ConvLayer(layers[-1], 3, allow_zero_in_degree=True)#, bias=False)#, k=2) #  , norm='both',

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        n = self.layers[1]
        if self.n_layers >= 2:
            h = self.conv2(g, h)
            h = F.relu(h)
            n = self.layers[2]
        if self.n_layers >= 3:
            h = self.conv3(g, h)
            h = F.relu(h)
            n = self.layers[3]
        if self.n_layers >= 4:
            h = self.conv4(g, h)
            h = F.relu(h)
            n = self.layers[4]
        if self.n_layers >= 5:
            h = self.conv5(g, h)
            h = F.relu(h)
            n = self.layers[5]
        if self.n_layers >= 6:
            h = self.conv6(g, h)
            h = F.relu(h)
            n = self.layers[6]
        if self.n_layers >= 7:
            h = self.conv7(g, h)
            h = F.relu(h)
            n = self.layers[7]
        if self.n_layers >= 8:
            h = self.conv8(g, h)
            h = F.relu(h)
            n = self.layers[8]
        if self.n_layers >= 9:
            h = self.conv9(g, h)
            h = F.relu(h)
            n = self.layers[9]
        h = nn.BatchNorm1d(n)(h)
        # nn.Dropout
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
from EarlyStopping import EarlyStopping
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

cv_train_losses=[[0],[0],[0],[0],[0]]
cv_val_losses=[[0],[0],[0],[0],[0]]

def train(g, model, n_epochs, metric_name, lr=5e-3, plot=False, val_split=4):
    global cv_train_losses
    global cv_val_losses
    global cv
    optimizer = torch.optim.Adam(model.parameters(), lr)
    # Get graph features, labels and masks
    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    cv_train_mask = g.ndata['cv_train_mask']
    cv_val_mask = g.ndata['cv_val_mask']

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=10, path="..\\ML_data\\ML_models_saves\\model," + ",".join([str(i) for i in layers[1:]]) +".pt")

    # Set sample importance weights
    weights = []
    # n0 = sum(x == 0 for x in labels[train_mask])
    n1 = sum(x == 1 for x in labels[train_mask])    # maby also adjust for cross validation weight size
    n2 = sum(x == 2 for x in labels[train_mask])
    train_losses=[]
    test_losses = []
    for e in range(n_epochs + 1):
        print("Epoch: ", e)
        # Forward
        logits = model(g, features)
        # Compute prediction
        pred = logits.argmax(1)
        # Compute loss
        # Note that you should only compute the losses of the nodes in the training set.
        weight = (torch.Tensor([0, n2, n1]))
        model.train()
        if cv:
            train_mask_obj # concatenate the splits into one object so that the loss can go over it at once
            val_mask_obj
            loss = nn.CrossEntropyLoss(weight)(logits[cv_train_mask].float(), labels[cv_train_mask].reshape(-1, ).long())
            cv_train_losses[val_split].append(loss)
            model.eval()
            val_loss = nn.CrossEntropyLoss(weight)(logits[cv_val_mask].float(), labels[cv_val_mask].reshape(-1, ).long())
            cv_val_losses[val_split].append(val_loss)

        else:
            loss = nn.CrossEntropyLoss(weight)(logits[train_mask].float(), labels[train_mask].reshape(-1, ).long())
            train_losses.append(loss)
            model.eval()
            val_loss = nn.CrossEntropyLoss(weight)(logits[val_mask].float(), labels[val_mask].reshape(-1, ).long())
            test_losses.append(val_loss)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        early_stopping(loss, model)
        #if early_stopping.early_stop:
        #    print("Early stopping")
        #    break

        if e % 5 == 0 or early_stopping.early_stop:
            # removing the impossible aa from the prediction
            met_pred, met_labels, met_train_mask, met_val_mask = slim_for_metrics(features, pred, labels, train_mask, val_mask)
            # Evaluation metric
            train_metric, val_metric = get_metric(met_pred, met_labels, met_train_mask, met_val_mask, metric_name)
            print('In epoch {}, loss: {:.3f}, train {} : {:.3f} , val {} : {:.3f}'.format(
                e, loss, metric_name, train_metric, metric_name, val_metric))
            if (early_stopping.early_stop or e == n_epochs):
                collection_variance_train.append(train_metric)
                collection_variance_val.append(val_metric)
                if plot:
                    train_losses_np = [t.detach().numpy() for t in train_losses]
                    test_losses_np = [t.detach().numpy() for t in test_losses]
                    fig, ax = plt.subplots()
                    ax.plot(train_losses_np, "b", label = "training loss")
                    ax.plot(test_losses_np, "r", label = "validation loss")
                    plt.xlabel("Epochs")
                    plt.ylabel("Loss")
                    ax.legend(loc="upper right")
                    plt.title("Graph neural network with early stopping\nLayers: " + ",".join([str(i) for i in layers]))
                    plt.savefig("..\\ML_data\\visualization\\Loss_curve," + ",".join([str(i) for i in layers]) +".png")
                    plt.show()
                break


    return np.array(pred[val_mask]), np.array(labels[val_mask])


# ## Baseline (onehot encoding)

# #### Chen Dataset (200 train , 51 validation)

# "In[323]:"
def training_cv(val_split):
    global layers
    dgl.seed(1)
    # Train the model
    layers = [g.ndata['feat'].shape[1],64,32,16,32,16,32,16,8]#,64,32,16,8,4] # , 64] # yannick
    print("model : ", layers)
    model = GCN(layers)
    pred, true = train(g, model, n_epochs=1000, metric_name="mcc", plot = False, val_split=val_split)
    return

val_splits=[0,1,2,3,4]
for i in val_splits:
    training_cv(i)

def training_variance(seed):
    global layers
    dgl.seed(seed)
    # Train the model
    layers = [g.ndata['feat'].shape[1],64,32,16,32,16,32,16,8]#,64,32,16,8,4] # , 64] # yannick
    print("model : ", layers)
    model = GCN(layers)
    pred, true = train(g, model, n_epochs=1000, metric_name="mcc", plot = False)
    return
collection_variance_train=[]
collection_variance_val=[]
for i in np.arange(20):
    training_variance(i)
print(collection_variance_train)
# [0.1113592462532849, 0.11181563008245886, 0.11645334559110472, 0.08997140821901614, 0.10329130651693717, 0.10667082145480833, 0.07715457209832315, 0.12132487195355517, 0.07790149408883712, 0.10461794167743783, 0.11801133411773122, 0.1319150389818283, 0.11237177824509738, 0.08966444851847302, 0.11487949769947974, 0.11984510226330194, 0.11823941709710761, 0.11647686238103433, 0.10276590072702023, 0.1089156634793606]
print(collection_variance_val)
# [0.11060733867311799, 0.1254872740076211, 0.09762422517451981, 0.11611058430434937, 0.10248519963170413, 0.11403688342683564, 0.09199460563136903, 0.12161389596996831, 0.08981922065363947, 0.1160582416501688, 0.12067975658070608, 0.11288906500383715, 0.1056797428521422, 0.10803676192359889, 0.10370016358214018, 0.10018139108096676, 0.09842886970273623, 0.10762924516005686, 0.11212984218450317, 0.11768801774825711]

# Train the model
layers = [g.ndata['feat'].shape[1],64,32,16,32,16,32,16,8]#,64,32,16,8,4] # , 64] # yannick
print("model : ", layers)
model = GCN(layers)
pred, true = train(g, model, n_epochs=1000, metric_name="mcc", plot = False)

import cProfile
cProfile.run("pred, true = train(g, model, n_epochs=500, metric_name='mcc')", "output.dat")
import pstats
from pstats import SortKey
with open("output_time.txt", "w")as f:
    p = pstats.Stats("output.dat", stream=f)
    p.sort_stats("time").print_stats()
with open("output_calls.txt", "w") as f:
    p = pstats.Stats("output.dat", stream=f)
    p.sort_stats("calls").print_stats()

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




