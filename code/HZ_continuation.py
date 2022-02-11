#!/usr/bin/env python
# coding: utf-8
import os
import time
import numpy as np
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

cv = True
cv_fold = 5

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

# if cv:
#     TRAIN_GRAPHS={**TRAIN_GRAPHS,**VAL_GRAPHS}


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

# if cv:
#     TRAIN_EMBS={**TRAIN_EMBS,**VAL_EMBS}

# "In[35]:"
import tensorflow as tf
total_feats = 0
graph_number = 0
train_graph_number = 0
train_graph_feats = 0
val_graph_number = 0
val_graph_feats = 0
test_graph_number = 0
test_graph_feats = 0
graph_lengths = {}
def create_dgl_graph(A, feats, labels, train, val, test, prot):
    g = dgl.from_scipy(A)
    feats = feats.reset_index(drop=True)
    g.ndata["feat"] = torch.tensor(feats[feats.columns].values).long()
    g.ndata["label"] = torch.tensor(labels).long()
    g.ndata["train_mask"] = torch.tensor(np.ones(A.shape[0]) == train)
    g.ndata["val_mask"]   = torch.tensor(np.ones(A.shape[0]) == val)
    g.ndata["test_mask"]  = torch.tensor(np.ones(A.shape[0]) == test)
    if test == 0:
        global graph_number
        global graph_lengths
        graph_lengths[graph_number] = [A.shape[0], prot]
        graph_number += 1
        global total_feats
        total_feats += labels.shape[0]
    if train == 1:
        global train_graph_number
        global train_graph_feats
        #g.ndata["graph_number"] = torch.full((labels.shape[0],1), graph_number)
        # global graph_lengths
        # graph_lengths[train_graph_number] = A.shape[0]
        train_graph_number += 1
        train_graph_feats += labels.shape[0]
    if val == 1:
        global val_graph_number
        global val_graph_feats
        #g.ndata["graph_number"] = torch.full((labels.shape[0],1), graph_number)
        # global graph_lengths
        # graph_lengths[train_graph_number] = A.shape[0]
        val_graph_number += 1
        val_graph_feats += labels.shape[0]
    if test == 1:
        global test_graph_number
        global test_graph_feats
        #g.ndata["graph_number"] = torch.full((labels.shape[0],1), graph_number)
        # global graph_lengths
        # graph_lengths[train_graph_number] = A.shape[0]
        test_graph_number += 1
        test_graph_feats += labels.shape[0]
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

        g1 = create_dgl_graph(A, feats, labels, train, val, test, prot)
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
def make_folder(folder):
    if os.path.exists(folder):
        print("folder already existing:", str(folder))
        shutil.rmtree(folder)
        os.makedirs(folder)
    else:
        os.makedirs(folder)


import torch

#my_graph = create_dgl_graph(A, feats, labels, 1, 0, 0)
#print(my_graph.ndata)

# "In[245]:"

create = True

train_embs_pth = os.path.join("../ML_data", "train_data", "graph")
val_embs_pth = os.path.join("../ML_data", "val_data", "graph")
test_embs_pth = os.path.join("../ML_data", "test_data", "graph")
cv_supplements_pth = os.path.join("../ML_data", "cv_supplements")
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

    pickle.dump([graph_lengths, graph_number, test_graph_feats], open(os.path.join(cv_supplements_pth), "wb"))
else:
    g_train = pickle.load(open(train_embs_pth, "rb"))
    g_val = pickle.load(open(val_embs_pth, "rb"))
    g_test = pickle.load(open(test_embs_pth, "rb"))
    cv_s = pickle.load(open(cv_supplements_pth, "rb"))
    graph_lengths, graph_number, test_graph_feats = cv_s[0], cv_s[1], cv_s[2]


g = dgl.batch([g_train, g_val, g_test])
del g_train
del g_val
del g_test



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

dgl.seed(1)

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
    return met_pred, met_labels, met_train_mask, met_val_mask, usable

from EarlyStopping import EarlyStopping
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

cv_train_losses=[[0],[0],[0],[0],[0]]
cv_val_losses=[[0],[0],[0],[0],[0]]

def train(g, model, n_epochs, metric_name, lr=1e-2, plot=False, val_split=4, cv_folds = 5):
    global cv_train_losses
    global cv_val_losses
    global cv
    global graph_number
    global graph_lengths
    global train_graph_feats
    global val_graph_feats
    global test_graph_feats
    name_position_dict = {}
    folds = []
    print("own counter training   : " + str(train_graph_feats))
    print("own counter validating : " + str(val_graph_feats))
    print("own counter testing    : " + str(test_graph_feats))
    print("dgl training on nodes  : " + str((g.ndata["train_mask"] == 1).sum()))
    print("dgl validating on nodes: " + str((g.ndata["val_mask"] == 1).sum()))
    print("dgl testing on nodes   : " + str((g.ndata["test_mask"] == 1).sum()))
    fold_size = graph_number / cv_folds
    counter = 0

    starting_index = -5
    lastend = 0
    testset=False
    for i in np.arange(cv_folds+1):
        start = round(i * fold_size)
        end = round((i+1) * fold_size)
        if i == cv_folds+1:
            testset = True
            end = graph_lengths.__sizeof__()
        set_length = 0
        for prot in np.arange(start, end):
            if not testset:
                set_length += (graph_lengths[prot])[0]
            if val_split == i and not testset:
                name_position_dict[counter] = [lastend, (lastend + (graph_lengths[prot])[0]), (graph_lengths[prot])[1], 0]
                lastend += graph_lengths[prot][0]
                counter += 1
            elif not testset:
                name_position_dict[counter] = [lastend, (lastend + (graph_lengths[prot])[0]), (graph_lengths[prot])[1], 1]
                lastend += graph_lengths[prot][0]
                counter += 1
            elif testset:
                name_position_dict[counter] = [lastend, (lastend + (graph_lengths[prot])[0]), (graph_lengths[prot])[1], 2]
                lastend += graph_lengths[prot][0]
                counter += 1
        if val_split == i:
            folds.append(torch.tensor(np.ones(set_length) == 0))
        else:
            folds.append(torch.tensor(np.ones(set_length) == 1))
    name_dict = os.path.join("../ML_data", "name_dict")
    pickle.dump(name_position_dict, open(os.path.join(name_dict),"wb"))
    train_mask = torch.cat(folds)
    val_mask = torch.tensor(train_mask==False)
    train_mask = torch.cat([train_mask,torch.tensor(np.zeros(test_graph_feats) > 0)])
    val_mask = torch.cat([val_mask, torch.tensor(np.zeros(test_graph_feats) > 0)])


    g.ndata["train_mask"] = train_mask
    g.ndata["val_mask"] = val_mask
    print("training on nodes: " + str((g.ndata["train_mask"] == 1).sum()))

    optimizer = torch.optim.Adam(model.parameters(), lr)
    # Get graph features, labels and masks
    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=10, min_delta=0.001, path="..\\ML_data\\ML_models_saves\\model," + ",".join([str(i) for i in layers[1:]]) +".pt")

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
        loss = nn.CrossEntropyLoss(weight)(logits[train_mask].float(), labels[train_mask].reshape(-1, ).long())
        model.eval()
        val_loss = nn.CrossEntropyLoss(weight)(logits[val_mask].float(), labels[val_mask].reshape(-1, ).long())
        if cv:
            cv_train_losses[val_split].append(loss)
            cv_val_losses[val_split].append(val_loss)

        else:
            train_losses.append(loss)
            test_losses.append(val_loss)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        early_stopping(loss, model)


        if e % 5 == 0 or early_stopping.early_stop:
            # removing the impossible aa from the prediction
#            met_pred, met_labels, met_train_mask, met_val_mask, usable = slim_for_metrics(features, pred, labels, train_mask, val_mask)
            # Evaluation metric # deppendict = {0: [0, 3, '2osu', 0], 1: [3, 6, '4ec9', 1], 2: [6, 9, '5uiq', 1], 3: [9, 12, '5o2c', 1], 4: [12, 15, '6fqm', 1]}
            pP_eval = pd.DataFrame(columns=["name", "train_metric", "val_metric"])
            train_metric_list = []
            val_metric_list = []
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
                if value[3] != 2: # not testing

                    temp_pred_df = pred_df.loc[start:end,:]
                    temp_labels_df = labels_df.loc[start:end, :]
                    temp_train_mask_df = train_mask_df.loc[start:end, :]
                    temp_val_mask_df = val_mask_df.loc[start:end, :]
                    pred_tp = torch.tensor(temp_pred_df.to_numpy())
                    labels_tp = torch.tensor(temp_labels_df.to_numpy())
                    train_mask_tp = torch.tensor(temp_train_mask_df.to_numpy())
                    val_mask_tp = torch.tensor(temp_val_mask_df.to_numpy())
                    #print("metric on: " + name)
                    #print("predictions:" + str(met_pred[start:end]))
                    #print("labels:" + str(met_labels[start:end]))
                    train_metric, val_metric = get_metric(pred_tp, labels_tp, train_mask_tp, val_mask_tp, metric_name)
                    if value[3] == 0:  # validation
                        val_metric_list.append(val_metric)
                    if value[3] == 1:  # training
                        train_metric_list.append(train_metric)
                    #print(*met_pred[start:end].numpy().flatten(), sep="")

#                    pP_eval = pP_eval.append({"name":name, "train_metric":train_metric, "val_metric":val_metric}, ignore_index=True)


#                print(*labels.numpy().flatten(), sep="")
#                print(*pred_tp.numpy().flatten(), sep="")
            # print("validation mcc mean: " + str(np.mean(pP_eval["val_metric"])))
            # print("validation standard deviation of the population: " + str(np.std(pP_eval["val_metric"], ddof=0)))
            # print("validation standard error of the population: " + str(np.std(pP_eval["val_metric"], ddof=0) / np.sqrt(np.size(pP_eval["val_metric"]))))
            # print("training mcc mean: " + str(np.mean(pP_eval["train_metric"])))
            # print("validation standard deviation of the population: " + str(np.std(pP_eval["train_metric"], ddof=0)))
            # print("validation standard error of the population: " + str(np.std(pP_eval["train_metric"], ddof=0) / np.sqrt(np.size(pP_eval["train_metric"]))))
            # print('In epoch {}, loss: {:.3f}, train {} : {:.3f} , val {} : {:.3f}'.format(e, loss, metric_name, train_metric, metric_name, val_metric))
            # print('In epoch {}, loss: {:.3f}, train mean {} : {:.3f} , val mean {} : {:.3f}'.format(e, loss, metric_name, np.mean(pP_eval["train_metric"]), metric_name, np.mean(pP_eval["val_metric"])))
            print('In epoch {}, loss: {:.3f}, train mean {} : {:.3f} , val mean {} : {:.3f}'.format(
                e, loss, metric_name, np.mean(train_metric_list), metric_name, np.mean(val_metric_list)))
            if (early_stopping.early_stop or e == n_epochs):
#                collection_variance_train.append(train_metric)
#                collection_variance_val.append(val_metric)
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

import GNN_architect

def training_cv(val_split):
    global layers
    dgl.seed(1)
    # Train the model
    layers = [g.ndata['feat'].shape[1],8,4]#,32,16,32,16,32,16,8] # , 64] # yannick
    kernel_size = [2, 1, 1]
    print("Split used for validation: " + str(val_split))
    print("model : ", layers)
    model = GNN_architect.GCN(layers, kernel_size)
    pred, true = train(g, model, n_epochs=1000, metric_name="mcc", plot = False, val_split=val_split)
    return

val_splits=[0,1,2,3,4]
for i in val_splits:
    training_cv(i)

#cv_train_losses=[[0,torch.tensor([1]),torch.tensor([1])],[0,torch.tensor([1]),torch.tensor([1])],[0,torch.tensor([1]),torch.tensor([1])],[0,torch.tensor([1]),torch.tensor([1])],[0,torch.tensor([1]),torch.tensor([1])]]
#cv_val_losses=[[0,torch.tensor([1]),torch.tensor([1])],[0,torch.tensor([1]),torch.tensor([1])],[0,torch.tensor([1]),torch.tensor([1])],[0,torch.tensor([1]),torch.tensor([1])],[0,torch.tensor([1]),torch.tensor([1])]]

fig, ax = plt.subplots()

for i, ent in enumerate(cv_train_losses):
    ent.pop(0)
    cv_train_losses[i] = [t.detach().numpy() for t in ent]
    ax.plot(cv_train_losses[i], "b", label="training loss")

for i, ent in enumerate(cv_val_losses):
    ent.pop(0)
    cv_val_losses[i] = [t.detach().numpy() for t in ent]
    ax.plot(cv_val_losses[i], "r", label="validation loss")

stderr_val = [value[-1] for value in cv_val_losses]
print("standard error of the sample: " + str(np.std(stderr_val, ddof=1) / np.sqrt(np.size(stderr_val))))
print("standard deviation of the sample: " + str(np.std(stderr_val, ddof=1)))


import matplotlib.patches as mpatches
plt.xlabel("Epochs")
plt.ylabel("Loss")
red_patch = mpatches.Patch(color='red', label='validation loss')
blue_patch = mpatches.Patch(color='blue', label='training loss')
legend = ax.legend()
legend.remove()
ax.legend(handles=[red_patch, blue_patch])
ax.legend(loc="upper right")
plt.title("Graph neural network with early stopping\nCross-validation test\nLayers: " + ",".join([str(i) for i in layers]))
plt.savefig("..\\ML_data\\visualization\\Cross-validation_Loss_curve," + ",".join([str(i) for i in layers]) +".png")
plt.show()

def training_variance(seed):
    global layers
    dgl.seed(seed)
    # Train the model
    layers = [g.ndata['feat'].shape[1],64,32,16,32,16,32,16,8]#,64,32,16,8,4] # , 64] # yannick
    print("model : ", layers)
    model = GNN_architect.GCN(layers)
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
model = GNN_architect.GCN(layers)
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
model = GNN_architect.GCN(layers)
pred, true = train(g, model, n_epochs=500, metric_name="mcc")

# "In[ ]:"




