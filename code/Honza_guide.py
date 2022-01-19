import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from Bio.PDB import *
# import spektral
import pickle
import dgl
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
import scipy
import shutil

import bio_embeddings.embed
#from bio_embeddings.embed import OneHotEncodingEmbedder
#from code.File_extractor import WD

#help(bio_embeddings.embed)


# Amino acid codons
STANDARD_AMINO_ACIDS={'ALA':'A', 'ARG':'R', 'ASN':'N', 'ASP':'D', 'CYS':'C', 'GLN':'Q', 'GLU':'E',
             'GLY':'G', 'HIS':'H', 'ILE':'I', 'LEU':'L',
                 'LYS':'K', 'MET':'M', 'PHE':'F', 'PRO':'P', 'SER':'S'
             , 'THR':'T', 'TRP':'W', 'TYR':'Y', 'VAL':'V'} #20


ACs=list(STANDARD_AMINO_ACIDS.keys())
AC_letters=list(STANDARD_AMINO_ACIDS.values())


# PDB parser
parser = PDBParser()

# SeqIO parser (getting sequences)
from Bio import SeqIO


# Bio embedders

# # Onehot embedding
# from bio_embeddings.embed.one_hot_encoding_embedder import OneHotEncodingEmbedder

# # SeqVec
# from bio_embeddings.embed.seqvec_embedder import SeqVecEmbedder

# # Glove
# from bio_embeddings.embed.glove_embedder import GloveEmbedder

# # Word2Vec
# from bio_embeddings.embed.word2vec_embedder import Word2VecEmbedder

# # BERT
# from bio_embeddings.embed.prottrans_bert_bfd_embedder import ProtTransBertBFDEmbedder

# # ESM
# from bio_embeddings.embed.esm_embedder import ESM1bEmbedder


#import bio_embeddings.embed.prottrans_xlnet_uniref100_embedder
#help(bio_embeddings.embed.prottrans_xlnet_uniref100_embedder.ProtTransXLNetUniRef100Embedder)

EMBEDDERS={"onehot":bio_embeddings.embed.OneHotEncodingEmbedder()}#,"seqvec":bio_embeddings.embed.SeqVecEmbedder(),
#          "glove":bio_embeddings.embed.GloveEmbedder()}

#embedder=bio_embeddings.embed.OneHotEncodingEmbedder()
#embedder.embed("MLSDKLSQD").shape
#embedder.embed("MLSDMLKSMFLKS").shape


# Useful functions

def get_sequence(residues):
    sequence = ''
    for res in residues:
        sequence += STANDARD_AMINO_ACIDS[res.get_resname()]

    return sequence


# Check if a residue is a ligand or not
def is_ligand(res):
    return res.get_full_id()[3][0] not in ["W", " "]


# Check if 2 residues are connected  (2 residues are connected if their alpha carbon atoms are close enough)
def are_connected(res1, res2, th):
    # Check all atoms in the first residue and get their coordinates
    for atom1 in res1.get_unpacked_list():
        coord1 = atom1.get_coord()
        # if some atom is a central carbon atom take its coordinates
        if atom1.get_name() == "CA":
            break

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

# Graph building
# This function parses a protein and returns
# 1. an array of the protein's adjacency matrix (residue nodes)
# 2. an embedding for the sequence of residues in the protein (features)
# 3. an array of labels representing ligandability of a residue to a ligand

def get_graph(data_folder, file):
    prot_file = os.path.join(data_folder, file)
    structure = parser.get_structure("My_protein", prot_file)

    # Get all residues
    residues = [res for res in structure.get_residues() if res.get_resname() in ACs]

    # Adjacency matrix at the residue level
    n_res = len(residues)
    A = np.zeros((n_res, n_res))
    for i in range(n_res):
        for j in range(n_res):
            A[i, j] = are_connected(residues[i], residues[j], th=6)  # Threshold = 6 Angstroms
    # Get all atoms of all ligands
    ligs_atoms = []
    n_ligands = 0
    for res in structure.get_residues():
        if is_ligand(res):
            n_ligands += 1
            ligs_atoms += res.get_unpacked_list()

    # Labels represent the ligandability of the residue
    n_res = len(residues)
    labels = np.zeros((n_res, 1))
    for i in range(n_res):
        labels[i] = is_ligandable(residues[i], ligs_atoms, th=4)  # Threshold = 4 Angstroms

    print(f"Number of ligands : {n_ligands}")
    print(int(np.sum(labels)), " of ", labels.shape[0], ' residue are labeled as 1')
    return scipy.sparse.csr_matrix(A), labels


def get_embeddings(data_folder, file, emb_name):
    prot_file = os.path.join(data_folder, file)
    structure = parser.get_structure("My_protein", prot_file)

    # Get all residues
    residues = [res for res in structure.get_residues() if res.get_resname() in ACs]
    # Get the protein sequence
    sequence = get_sequence(residues)

    # Get the sequence embeddings
    embedding = EMBEDDERS[emb_name].embed(sequence)

    if emb_name == "onehot":
        emb1 = EMBEDDERS[emb_name].embed(sequence[0]).reshape(1, -1)
        for i in range(1, len(sequence)):
            emb2 = EMBEDDERS[emb_name].embed(sequence[i]).reshape(1, -1)
            emb1 = np.concatenate((emb1, emb2), axis=0)
        embedding = emb1
    return embedding

from dgl.nn import SGConv as ConvLayer

datasets=["chen dataset","holo4k",]
# dataset="chen dataset"
dataset="holo4k"
data_folder='./p2rank-datasets/'+dataset # this is the folder where the pdb files are

def get_sequence_from_file(data_folder, file):
    prot_file = os.path.join(data_folder, file)
    structure = parser.get_structure("My_protein", prot_file)

    # Get all residues
    residues = [res for res in structure.get_residues() if res.get_resname() in ACs]
    # Get the protein sequence
    return get_sequence(residues)

# Define training and validation data
n_data=len(os.listdir(data_folder))
n_train=int(0.8*n_data)
np.random.seed(16)
train_indices=np.random.choice(n_data,n_train,replace=False)

trainset=[os.listdir(data_folder)[i] for i in train_indices]
valset=[os.listdir(data_folder)[i] for i in range(n_data) if i not in train_indices]

def make_folder(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

train_graphs=os.path.join(data_folder,"train_data","graphs")
val_graphs=os.path.join(data_folder,"val_data","graphs")

make_folder(train_graphs)
make_folder(val_graphs)



# Graph building

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


obj1 = PDBParser().get_structure("1ai2", "C:\\Users\\administ\\PycharmProjects\\Phosphorylation\\data\\ser90\\wP\\wP_paired\\dali\\1ai2.txt")

# removes hetatm ligands and returns the one letter code as an array
# requires an object file from the PDBParser().get_structure("name", "path")
def remhetatm(pdbstructure):
    ress = []
    for res in pdbstructure.get_residues():
        if res.resname in STANDARD_AMINO_ACIDS.keys():
            ress.append(STANDARD_AMINO_ACIDS[res.resname])
    return ress

#embedder.embed("".join(remhetatm(obj1)))

# TODO
# attach the onehotencoding feature to the atom in the graph

# to retrieve the one hot encoding embeddings
get_embeddings("C:\\Users\\administ\\PycharmProjects\\Phosphorylation\\data\\ser90\\wP\\wP_paired\\dali", "1ai2.txt", "onehot")

import pickle as pkl
from pathlib import Path
WD = Path(__file__).resolve().parents[1]
ld = pkl.load(open(WD / "phos_info" / "structure_locations.txt", "rb"))
for key, val in ld.items():
    get_graph((WD/"pdb_collection"), ((key +".txt")))

print("test")


