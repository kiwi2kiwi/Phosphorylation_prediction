#!/usr/bin/env python
# coding: utf-8
import os
import time
import numpy as np
import pandas as pd

import pickle
import dgl
import dgl.nn
import scipy
import shutil

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
# from bio_embeddings.embed import OneHotEncodingEmbedder # yannick

# # SeqVec
# from bio_embeddings.embed.seqvec_embedder import SeqVecEmbedder

# # Glove working
import nltk
nltk.download('omw-1.4')
from bio_embeddings.embed.glove_embedder import GloveEmbedder

# # Word2Vec
# from bio_embeddings.embed.word2vec_embedder import Word2VecEmbedder

# # BERT
# from bio_embeddings.embed.prottrans_bert_bfd_embedder import ProtTransBertBFDEmbedder

# # ESM
# from bio_embeddings.embed.esm_embedder import ESM1bEmbedder


#import bio_embeddings.embed.prottrans_xlnet_uniref100_embedder
#help(bio_embeddings.embed.prottrans_xlnet_uniref100_embedder.ProtTransXLNetUniRef100Embedder)

EMBEDDERS={#"onehot":OneHotEncodingEmbedder(),#"seqvec":SeqVecEmbedder(),
          "glove":GloveEmbedder()}#, "bert":ProtTransBertBFDEmbedder()}


# embedder = OneHotEncodingEmbedder()
# embedder.embed("MLSDKLSQD").shape
# embedder.embed("MLSDMLKSMFLKS").shape

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

# TODO change lines 110 and 111 to switch between contact and distance map
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
                            # A[i_num, j_num] = 1 # use this for a contact map
                            A[i_num, j_num] = th - distance # use this for a distance map. Closer atoms get higher score
    return A


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
    # TODO change the threshold here
    A = are_connected(A, residues, th=6)  # Threshold = 6 Angstroms
    labels = np.zeros((n_res,1))

    # for idx, residue_structured in enumerate(residues):
    #     if label_dict[file[:-4]].__contains__(float(residue_structured.id[1])):
    #         labels[idx] = 2
    #         true_labels += 1
    #     elif residue_structured.get_resname() in ["THR", "TYR", "SER"]:
    #         labels[idx] = 1
    #         false_labels += 1
    #     else:
    #         impossible_labels += 1

    return scipy.sparse.csr_matrix(A), labels


from pathlib import Path
WD = Path().resolve().parents[0]
import h5py
embedding_dictionary = {}
embedding_source = "external"
embedding_source = "internal"

# TODO import external embeddings here
# create a file that looks like the bert embedder output. This can be used as custom embeddings
if embedding_source == "external":
    data = np.load(WD / "ML_data" / "external_embeddigs" / "phospho_bert_emb.npy", allow_pickle=True)
    for i in data:
        embedding_dictionary[i[0]] = i[1]

def get_embeddings(data_folder, file, emb_name):
    prot_file = os.path.join(data_folder, file)
    structure = parser.get_structure(file[:-4], prot_file)

    # Get all residues
    residues = [res for res in structure.get_residues() if res.get_resname() in ACs]
    valid = []
    for res in residues:
        if res.get_resname() in ["SER","THR","TYR"]:
            valid.append(1)
        else:
            valid.append(0)
    #valid = [1 for res in structure.get_residues() if res.get_resname() in ["SER","THR","TYR"]]
    # Get the protein sequence
    sequence,index_list = get_sequence(residues)
    # Get the sequence embeddings
    #embedding1 = EMBEDDERS["onehot"].embed(sequence)
    global embedding_source
    if embedding_source == "external":
        global embedding_dictionary
        embedding = embedding_dictionary[file[:-4]]
    else:
        embedding = EMBEDDERS[emb_name].embed(sequence)
    embedding = pd.DataFrame(embedding)
    embedding[-1] = valid
    embedding[""] = index_list
    embedding = embedding.set_index("")
    embedding.index.name = None

    if emb_name == "onehot":
        embedding = embedding_twister(sequence, emb_name)
    return embedding

# yannick
import pickle as pkl
from pathlib import Path
WD = Path().resolve().parents[0]

label_dict = {}
with open(WD / "phos_info" / "info_on_new_pdb_phos.txt","r") as iop:
    for line in iop.readlines():
        line = line.split()
        locations = [float(loc) for loc in line[2].split(",")]
        label_dict[line[0]] = locations

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


dataset = "holo4k"
data_folder = "../ML_data/parsed_new_pdbs" # yannick

create_fasta = False
if create_fasta:
    with open(os.path.join("..\\ML_data\\phospho.fasta"), "w") as fasta_file:
        to_write = ""
        for file in os.listdir(data_folder):
            prot_file = os.path.join(data_folder, file)
            structure = parser.get_structure(file[:-4], prot_file)
            # Get all residues
            residues = [res for res in structure.get_residues() if res.get_resname() in ACs]
            residues_one_letter = [STANDARD_AMINO_ACIDS[res.get_resname()] for res in residues]
            residues_one_letter = "".join(residues_one_letter)
            to_write = (str(to_write)+str(">")+str(file[:-4])+"\n"+str(residues_one_letter)+"\n")
        fasta_file.writelines(to_write)
print("fasta written")


# "In[15]:"


# ## Dealing with residues with zero alpha carbon atoms

# "In[77]:"
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

# Define training and validation data
n_data = sum(1 for line in open(WD / "phos_info" / "info_on_new_pdb_phos.txt"))

predictionset = [os.listdir(data_folder)[i] for i in np.arange(n_data)]

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
cv_seq = os.path.join("D:/data", dataset, "cv_data", "sequences")

# "In[18]:"


# make_folder(train_seq)
# make_folder(val_seq)


# ### Graphs

# "In[21]:"
prediction_graphs = os.path.join("../ML_data", "prediction", "graphs")
print(f"Prediction graphs : {n_data} elements")





# Create graphs

def graph_generation(index_set, save_path):
    for file in index_set:
        print(f"processing file {file}  ", file, "  ....")
        # get graph and labels
        results = get_graph(data_folder, file)
        pickle.dump(results, open(os.path.join(save_path, file[:-4] + ".p"), "wb"))


def generate_graphs():
    make_folder(prediction_graphs)
    graph_generation(predictionset, prediction_graphs)


generate_graphs()

# ### Embeddings

# "In[33]:"


emb_name = "glove"
prediction_embs = os.path.join("../ML_data", "prediction", "embeddings", emb_name)

# "In[349]:"



# "In[350]:"


print(f"Prediction embeddings : {n_data} elements")

# Create embeddings
def embedding_generation(index_set, save_path):
    for file in index_set:
        print(f"processing file {file}  ", file, "  ....")
        # Get embeddings
        results = get_embeddings(data_folder, file, emb_name)
        pickle.dump(results, open(os.path.join(save_path, file[:-4] + ".p"), "wb"))


def create_embeddings():
    make_folder(prediction_embs)
    embedding_generation(predictionset, prediction_embs)
create_embeddings()

print("done")

# "In[22]:"

