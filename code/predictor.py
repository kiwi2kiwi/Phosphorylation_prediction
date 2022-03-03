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


import warnings

warnings.filterwarnings("ignore")

# supplements from previous script
emb_name = "bert"


# Load the graphs
from pathlib import Path
WD = Path(__file__).resolve().parents[1]
import pickle

PREDICTION_GRAPHS = {}
prediction_graphs = WD / "ML_data" / "prediction" / "graphs"

for file in os.listdir(prediction_graphs):
    PREDICTION_GRAPHS[file[:-2]] = pickle.load(open(os.path.join(prediction_graphs, file), "rb"))

# Load the embeddings

prediction_embs = WD / "ML_data" / "prediction" / "embeddings" / emb_name
PREDICTION_EMBS = {}
for file in os.listdir(prediction_embs):
    PREDICTION_EMBS[file[:-2]] = pickle.load(open(os.path.join(prediction_embs, file), "rb"))

print(f"Training embeddings loaded : {len(PREDICTION_EMBS)} items")
# "In[35]:"
# import tensorflow as tf
# total_feats = 0
# graph_number = 0
# train_graph_number = 0
# train_graph_feats = 0
# val_graph_number = 0
# val_graph_feats = 0
# test_graph_number = 0
# test_graph_feats = 0
# graph_lengths = {}
def create_dgl_graph(A, feats, labels, train, val, test, prot):
    g = dgl.from_scipy(A)
    feats = feats.reset_index(drop=True)
    g.ndata["feat"] = torch.tensor(feats[feats.columns].values).long()
    g.ndata["feat2"] = torch.tensor(feats[feats.columns].values)
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

prediction_embs_pth = os.path.join("../ML_data", "prediction", "graph")
onehot = False
print("Creating training set...")
g = create_dgl_data(PREDICTION_GRAPHS, PREDICTION_EMBS, train=1, val=0, test=0, onehot=onehot)



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


def predictor(g, model):

    # Get graph features, labels and masks
    features = g.ndata['feat']

    df_feat = pd.DataFrame(features.numpy())
    usable = df_feat[df_feat.shape[1]-1] == 1
    # Forward
    model.eval()
    logits = model(g, features)
    # Compute prediction
    pred = logits.argmax(1)
    #usable = usable.iloc[:,0]
    boolean_list_negative = usable==False
    pred[boolean_list_negative] = 0
    print((torch.tensor([1, 2, 1, 2]) == 2).nonzero(as_tuple=True)[0].detach().numpy())
    with open(WD / "phos_info" / "info_on_new_pdb_phos.txt", "r") as iop:
        line = iop.readline()
        line = line.split()
        modelid = float(line[5])
        chainid = line[1]
        availables = list(map(float,line[4].split(",")))
    for idx, i in enumerate(pred):
        if i == 0 and boolean_list_negative[idx] == False:
            pred[idx] = 1
    result_possibles = (pred != 0)
    ava = torch.tensor(availables)
    to_select = ava[(pred[result_possibles] == 2)].detach().numpy()
    to_select_negative = ava[(pred[result_possibles] == 1)].detach().numpy()
    indices_sel = "[" + ",".join(map(str, map(int, to_select))) + "]"
    indices_sel_negative = "[" + ",".join(map(str, map(int, to_select_negative))) + "]"
    print("the red residues are predicted to be phosphorylated, and the blue ones are unphosphorylated THR, TYR, SER")
    print("load the protein and paste this into the pymol console to see the predicted residues")
    print("show surface")
    print("color green")#, 7tvs")
    print("select resi "+indices_sel)
    print("color red, sele")
    print("select resi "+indices_sel_negative)
    print("color blue, sele")
    print(pred)




import GNN_architect

dgl.seed(1)
# model,8,4.pt
model_name = "model,16,8,8,8val_split0glove.pt"#"model,2048,1024,512,256,128,64,32val_split0.pt"#"model,512,256,128,64,32val_split0.pt"
#model_name = "model,8val_split0.pt"
modelpath = WD / "ML_data" / "ML_models_saves" / model_name
model = pickle.load(open(modelpath,"rb"))
#GNN_architect.GCN(layers, kernel_size)
predictor(g, model)


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

