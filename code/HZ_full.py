#!/usr/bin/env python
# coding: utf-8
import os
import time
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd

# import spektral
import pickle
import dgl
import dgl.nn
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
import scipy
import shutil

#from code.File_extractor import WD

# help(bio_embeddings.embed)

import warnings

warnings.filterwarnings("ignore")

# Amino acid codons
STANDARD_AMINO_ACIDS={'ALA':'A', 'ARG':'R', 'ASN':'N', 'ASP':'D', 'CYS':'C', 'GLN':'Q', 'GLU':'E',
             'GLY':'G', 'HIS':'H', 'ILE':'I', 'LEU':'L',
                 'LYS':'K', 'MET':'M', 'PHE':'F', 'PRO':'P', 'SER':'S'
             , 'THR':'T', 'TRP':'W', 'TYR':'Y', 'VAL':'V'} #20


ACs=list(STANDARD_AMINO_ACIDS.keys())
AC_letters=list(STANDARD_AMINO_ACIDS.values())

from Bio.PDB import *

# PDB parser
parser = PDBParser()

# SeqIO parser (getting sequences)
from Bio import SeqIO


# Bio embedders

# # Onehot embedding working
# from bio_embeddings.embed.one_hot_encoding_embedder import OneHotEncodingEmbedder
from bio_embeddings.embed import OneHotEncodingEmbedder # yannick

# # SeqVec
# from bio_embeddings.embed.seqvec_embedder import SeqVecEmbedder

# # Glove working
from bio_embeddings.embed.glove_embedder import GloveEmbedder

# # Word2Vec
# from bio_embeddings.embed.word2vec_embedder import Word2VecEmbedder

# # BERT
# from bio_embeddings.embed.prottrans_bert_bfd_embedder import ProtTransBertBFDEmbedder

# # ESM
# from bio_embeddings.embed.esm_embedder import ESM1bEmbedder


#import bio_embeddings.embed.prottrans_xlnet_uniref100_embedder
#help(bio_embeddings.embed.prottrans_xlnet_uniref100_embedder.ProtTransXLNetUniRef100Embedder)

EMBEDDERS={"onehot":OneHotEncodingEmbedder(),#"seqvec":SeqVecEmbedder(),
          "glove":GloveEmbedder()}#, "bert":ProtTransBertBFDEmbedder()}
# yannick
embedder1 = GloveEmbedder()
embedder = OneHotEncodingEmbedder()
print(embedder1.embed("MLSD").shape)
print(embedder.embed("MLSD").shape)

embedder = OneHotEncodingEmbedder()
embedder.embed("MLSDKLSQD").shape
embedder.embed("MLSDMLKSMFLKS").shape

true_labels = 0
false_labels = 0
impossible_labels = 0

# ## Graph connectivity and ligandability

# "In[11]:"


# Useful functions

def get_sequence(residues):
    sequence = ''
    index_list = []
    for res in residues:
        sequence += STANDARD_AMINO_ACIDS[res.get_resname()]
        index_list.append(res.id[1])

    return sequence,index_list


# Check if a residue is a ligand or not
def is_ligand(res):
    return res.get_full_id()[3][0] not in ["W", " "]


# Check if 2 residues are connected  (2 residues are connected if their alpha carbon atoms are close enough)
# recieves a numpy 2D array (A), unpacked residue list of the residues, and a threshold
def are_connected(A, residues, th):
    for i_num, i in enumerate(residues):
        atm1 = i.child_list[0].get_coord()
        for j_num, j in enumerate(residues):
            atm2 = j.child_list[0].get_coord()
            if abs(atm1[0] - atm2[0]) <= th:
                if abs(atm1[1] - atm2[1]) <= th:
                    if abs(atm1[2] - atm2[2]) <= th:
                        distance = np.linalg.norm(atm1 - atm2)
                        if distance < th:
                            A[i_num, j_num] = 1
    return A
    # Check all atoms in the first residue and get their coordinates
    # for atom1 in res1.get_unpacked_list():
    #     coord1 = atom1.get_coord()
    #     # if some atom is a central carbon atom take its coordinates
    #     if atom1.get_name() == "CA":
    #         break
    #
    # for atom2 in res2.get_unpacked_list():
    #     coord2 = atom2.get_coord()
    #     # if some atom is a central carbon atom take its coordinates
    #     if atom2.get_name() == "CA":
    #         break

    # Check distance
    # distance = np.linalg.norm(coord1 - coord2)
    # if distance < th:
    #     return 1



# Check ligandability of a residue
def is_ligandable(res, ligs_atoms, th):
    for atom in res.get_unpacked_list():
        atom_coord = atom.get_coord()
        for lig_atom in ligs_atoms:
            lig_coord = lig_atom.get_coord()
            distance = np.linalg.norm(lig_coord - atom_coord)
            if distance < th:
                return 1
    return 0


# ## Graphs, Labels and Embeddings

# "In[167]:"


# This function parses a protein and returns
# 1. an array of the protein's adjacency matrix (residue nodes)
# 2. an embedding for the sequence of residues in the protein (features)
# 3. an array of labels representing ligandability of a residue to a ligand

def embedding_twister(sequence, emb_name):
    emb1 = EMBEDDERS[emb_name].embed(sequence[0]).reshape(1, -1)
    for i in range(1, len(sequence)):
        emb2 = EMBEDDERS[emb_name].embed(sequence[i]).reshape(1, -1)
        emb1 = np.concatenate((emb1, emb2), axis=0)
    embedding = emb1
    return embedding


def get_graph(data_folder, file):
    global true_labels
    global false_labels
    global impossible_labels
    prot_file = os.path.join(data_folder, file)
    structure = parser.get_structure(file[:-4], prot_file)

    # Get all residues
    residues = [res for res in structure.get_residues() if res.get_resname() in ACs]

    # Adjacency matrix at the residue level
    n_res = len(residues)
    A = np.zeros((n_res, n_res))
    A = are_connected(A, residues, th=6)  # Threshold = 6 Angstroms
    labels = np.zeros((n_res,1))

    for idx, residue_structured in enumerate(residues):
        if label_dict[file[:-4]].__contains__(float(residue_structured.id[1])):
            labels[idx] = 2
            true_labels += 1
        elif residue_structured.get_resname() in ["THR", "TYR", "SER"]:
            labels[idx] = 1
            false_labels += 1
        else:
            impossible_labels += 1

    return scipy.sparse.csr_matrix(A), labels


def get_embeddings(data_folder, file, emb_name):
    prot_file = os.path.join(data_folder, file)
    structure = parser.get_structure(file[:-4], prot_file)

    # Get all residues
    residues = [res for res in structure.get_residues() if res.get_resname() in ACs]
    valid = []
    for res in structure.get_residues():
        if res.get_resname() in ["SER","THR","TYR"]:
            valid.append([1])
        else:
            valid.append(0)
    #valid = [1 for res in structure.get_residues() if res.get_resname() in ["SER","THR","TYR"]]
    # Get the protein sequence
    sequence,index_list = get_sequence(residues)

    # Get the sequence embeddings
    #embedding1 = EMBEDDERS["onehot"].embed(sequence)
    embedding = EMBEDDERS[emb_name].embed(sequence)
    embedding = pd.DataFrame(embedding)
    embedding[512] = valid
    embedding[""] = index_list
    embedding = embedding.set_index("")
    embedding.index.name = None

    if emb_name == "onehot":
        embedding = embedding_twister(sequence, emb_name)
    return embedding

# yannick
import pickle as pkl
from pathlib import Path
WD = Path(__file__).resolve().parents[1]

label_dict = {}
with open(WD / "phos_info" / "info_on_phos.txt","r") as iop:
    for line in iop.readlines():
        line = line.split()
        locations = [float(loc) for loc in line[2].split(",")]
        label_dict[line[0]] = locations

label_paths = pkl.load(open(WD / "phos_info" / "structure_locations.txt", "rb"))


# "In[5]:"


def get_sequence_from_file(data_folder, file):
    prot_file = os.path.join(data_folder, file)
    structure = parser.get_structure("My_protein", prot_file)

    # Get all residues
    residues = [res for res in structure.get_residues() if res.get_resname() in ACs]
    # Get the protein sequence
    return get_sequence(residues)[0]


# ## Dataset

# "In[20]:"


datasets = ["chen dataset", "holo4k", ]
datasets = [""] # yannick
# dataset="chen dataset"
dataset = "holo4k"
data_folder = './p2rank-datasets/' + dataset
data_folder = "../pdb_collection" # yannick
# "In[15]:"


# ## Dealing with residues with zero alpha carbon atoms

# "In[77]:"


def has_zero(res):
    for atom1 in res.get_unpacked_list():
        # if some atom is a central carbon atom take its coordinates
        if atom1.get_name() == "CA":
            return False
    return True


# Does the protein have a residue which has a 0 CA
def has_zero_CA(data_folder, file):
    prot_file = os.path.join(data_folder, file)
    structure = parser.get_structure("My_protein", prot_file)

    # Get all residues
    for res in structure.get_residues():
        if res.get_resname() in ACs:
            if has_zero(res):
                return True
    return False


def zero_ca_preparation():
    # List of proteins that have 0 CA
    zero_ca = []
    n = 0
    for file in os.listdir(data_folder):
        n += 1
        if has_zero_CA(data_folder, file):
            zero_ca.append(file)
        if n % 20 == 0 and n != 0:
            print(n, "files processed")
    print(n, "Total files processed")
    print(len(zero_ca), "have some res of 0 CA")
    pdb_files = n
pdb_files = 360
# "In[78]:"


def connected(atom1, res2, th=6):
    coord1 = atom1.get_coord()
    for atom2 in res2.get_unpacked_list():
        coord2 = atom2.get_coord()
        # if some atom is a central carbon atom take its coordinates
        if atom2.get_name() == "CA":
            break

    # Check distance
    distance = np.linalg.norm(coord1 - coord2)
    if distance < th:
        return 1

    return 0


# "In[79]:"


import random
def zero_ca():
    print("Dataset :", dataset)
    print("proteins which contain at least a residue of 0 CA :", len(zero_ca))
    for file in zero_ca:
        prot_file = os.path.join(data_folder, file)
        structure = parser.get_structure("My_protein", prot_file)
        non_zero_res = []
        zero_res = []
        for res in structure.get_residues():
            if res.get_resname() in ACs:
                if has_zero(res):
                    zero_res.append(res)
                else:
                    non_zero_res.append(res)
        # Check if the connectivity is the same for each atom of the zero CA residue
        for res1 in zero_res:
            for res2 in non_zero_res:
                start = connected(res1.get_unpacked_list()[0], res2)
                for atom1 in res1.get_unpacked_list():
                    end = connected(atom1, res2)
                    if start != end:
                        print("error in protein !!", file, "  residue", res1.get_id(), "with residue", res2.get_id())
                        print("Non zero", len(non_zero_res), "zero", len(zero_res))


# zero_ca_preparation()
# zero_ca()


# ## Create training and validation sets

# "In[16]:"


# Define training and validation data
n_data = len(os.listdir(data_folder))
np.random.seed(16)
indices = np.arange(n_data)
np.random.shuffle(indices)
n_train = int(0.7 * n_data)
n_val = int(0.2 * n_data)
n_test = int(n_data-n_val-n_train)
train_indices = indices[:n_train]
val_indices = indices[n_train:n_train+n_val]
test_indices = indices[n_train+n_val:]

trainset = [os.listdir(data_folder)[i] for i in train_indices]
valset = [os.listdir(data_folder)[i] for i in val_indices]
testset = [os.listdir(data_folder)[i] for i in test_indices]


def make_folder(folder):
    if os.path.exists(folder):
        print("folder already existing:", str(folder))
        shutil.rmtree(folder)
        os.makedirs(folder)
    else:
        os.makedirs(folder)


# ### Sequences

# "In[14]:"


import warnings

warnings.filterwarnings("ignore")

# "In[16]:"


train_seq = os.path.join("D:/data", dataset, "train_data", "sequences")
val_seq = os.path.join("D:/data", dataset, "val_data", "sequences")
test_seq = os.path.join("D:/data", dataset, "test_data", "sequences")

# "In[18]:"


# make_folder(train_seq)
# make_folder(val_seq)

# "In[19]:"


# for file in trainset:
#     sequence = get_sequence_from_file(data_folder, file)
#     with open(os.path.join(train_seq, file[:-4] + ".txt"), "w") as f:
#         f.write(sequence)
# for file in valset:
#     sequence = get_sequence_from_file(data_folder, file)
#     with open(os.path.join(val_seq, file[:-4] + ".txt"), "w") as f:
#         f.write(sequence)

# ### Graphs

# "In[21]:"
train_graphs = os.path.join("../ML_data", "train_data", "graphs")
val_graphs = os.path.join("../ML_data", "val_data", "graphs")
test_graphs = os.path.join("../ML_data", "test_data", "graphs")
# "In[330]:"
# "In[336]:"
print(f"Training graphs : {n_train} elements")
print(f"Validation graphs : {n_val} elements")
print(f"Testing graphs : {n_test} elements")

training_set_intervall_size = 20





# Create graphs

def graph_generation(n, index_set, set_type, save_path):
    a = n * training_set_intervall_size
    b = (n + 1) * training_set_intervall_size
    print(f"{set_type} graphs : from {a} to {b}")
    i = a
    for file in index_set[a:b]:
        i += 1
        print(f"processing file {i}  ", file, "  ....")
        # get graph and labels
        results = get_graph(data_folder, file)
        pickle.dump(results, open(os.path.join(save_path, file[:-4] + ".p"), "wb"))


def generate_graphs():
    # make_folder(train_graphs)
    # make_folder(val_graphs)
    # make_folder(test_graphs)
    # "In[339]:"
    batch_number = 0
    while pdb_files > (batch_number * training_set_intervall_size):
        graph_generation(batch_number, trainset, "Training", train_graphs)
        graph_generation(batch_number, valset, "Validation", val_graphs)
        graph_generation(batch_number, testset, "Testing", test_graphs)
        batch_number += 1


generate_graphs()

# ### Embeddings

# "In[33]:"


emb_name = "glove"
train_embs = os.path.join("../ML_data", "train_data", "embeddings", emb_name)
val_embs = os.path.join("../ML_data", "val_data", "embeddings", emb_name)
test_embs = os.path.join("../ML_data", "test_data", "embeddings", emb_name)

# "In[349]:"



# "In[350]:"


print(f"Training embeddings : {n_train} elements")
print(f"Validation embeddings : {n_val} elements")
print(f"Testing embeddings : {n_test} elements")


# Create embeddings
def embedding_generation(n, index_set, set_type, save_path):
    a = n * training_set_intervall_size
    b = (n + 1) * training_set_intervall_size
    print(f"{set_type} embeddings : from {a} to {b}")
    i = a
    for file in index_set[a:b]:
        i += 1
        print(f"processing file {i}  ", file, "  ....")
        # Get embeddings
        results = get_embeddings(data_folder, file, emb_name)
        pickle.dump(results, open(os.path.join(save_path, file[:-4] + ".p"), "wb"))


def create_embeddings():
    # make_folder(train_embs)
    # make_folder(val_embs)
    # make_folder(test_embs)
    # "In[351]:"
    batch_number = 0
    while pdb_files > (batch_number * training_set_intervall_size):
        embedding_generation(batch_number, trainset, "Training", train_embs)
        embedding_generation(batch_number, valset, "Validation", val_embs)
        embedding_generation(batch_number, testset, "Testing", test_embs)
        batch_number += 1

create_embeddings()


# "In[22]:"

