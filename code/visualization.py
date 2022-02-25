from pathlib import Path
import pickle
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import os
import pandas as pd
from Bio.PDB import *
parser = PDBParser()

one_code = {"S": 0,
            "Y": 1,
            "T": 2,
            }
STANDARD_AMINO_ACIDS={'ALA':'A', 'ARG':'R', 'ASN':'N', 'ASP':'D', 'CYS':'C', 'GLN':'Q', 'GLU':'E',
             'GLY':'G', 'HIS':'H', 'ILE':'I', 'LEU':'L',
                 'LYS':'K', 'MET':'M', 'PHE':'F', 'PRO':'P', 'SER':'S'
             , 'THR':'T', 'TRP':'W', 'TYR':'Y', 'VAL':'V'} #20


ACs=list(STANDARD_AMINO_ACIDS.keys())
AC_letters=list(STANDARD_AMINO_ACIDS.values())


np.random.RandomState(1)
emb_name = "bert"
emb_folder = os.path.join("../ML_data", "train_data", "embeddings", emb_name)

sample_n = len(os.listdir(emb_folder))
def combine_training_embeddings():
    index_dict = {}
    res_dict = {}
    WD = Path(__file__).resolve().parents[1]
    with open(WD / "phos_info" / "info_on_phos.txt", "r") as iop:
        for line in iop.readlines():
            line = line.split()
            structure = parser.get_structure(line[0], (WD / "pdb_collection" / (line[0]+".txt")))
            residues = [res for res in structure.get_residues() if res.get_resname() in ACs]
            res_dict[line[0]]=residues
            trues = line[2].split(",")
            trues = list(map(float, trues))
            possibles = line[6].split(",")
            possibles = list(map(float, possibles))
            index_dict[line[0]] = [trues, possibles]

    training_embs = pd.DataFrame()
    counter = 0
    for file in os.listdir(emb_folder):
        if counter == 208:
            print("stop")
        to_append = pd.DataFrame(pickle.load(open(os.path.join(emb_folder, file), "rb")))
        label_list = pd.Series(0, to_append.index)
        true_labels = index_dict[file[:-2]][0]
        label_list.loc[true_labels] = 1
        to_append[1024] = label_list
        possible_list = pd.Series(0, to_append.index)
        possible_labels = index_dict[file[:-2]][1]
        temp = possible_labels
        for i in temp:
            if i>to_append.index[-1]:
                possible_labels.remove(i)
        possible_list.loc[possible_labels] = 1
        to_append[1025] = possible_list

        STW_labels = []
        for i in res_dict[file[:-2]]:
            if STANDARD_AMINO_ACIDS[i.resname] in one_code.keys():
                STW_labels.append(one_code[STANDARD_AMINO_ACIDS[i.resname]])
            else:
                STW_labels.append(3)
        STW_list = pd.Series(STW_labels, to_append.index)
        to_append[1026] = STW_list

        if to_append[(to_append[1024] == 1) & (to_append[1024] == 0)].shape[0] > 0:
            print("stop")
        to_append = to_append[to_append[1025] == 1]
        to_append = to_append.reset_index(drop=True)
        print(counter)
        training_embs = training_embs.append(to_append)
        counter +=1
    training_embs = training_embs.reset_index(drop=True)
    print(training_embs.shape)
    pickle.dump(training_embs, open(os.path.join("../ML_data", "visualization", "tsne_embs.txt"), "wb"))
    return training_embs

training_embs = combine_training_embeddings()
training_embs = pickle.load(open(os.path.join("../ML_data", "visualization", "tsne_embs.txt"), "rb"))

false_labels = training_embs.loc[training_embs[1024] == 0]
false_labels = false_labels.reset_index(drop=True)
true_labels = training_embs.loc[training_embs[1024] == 1]
true_labels = true_labels.reset_index(drop=True)
false_labels = false_labels.sample(frac=(true_labels.shape[0]/false_labels.shape[0])*10)
sample_n = true_labels.shape[0]
tsne_data = pd.DataFrame()
tsne_data = tsne_data.append(true_labels)
tsne_data = tsne_data.append(false_labels)
X = tsne_data.iloc[:,:-3]
Y = tsne_data.iloc[:,-3:]
print("numper of phosphosites: " + str(sample_n))


perplex = round((X.shape[0])*0.01)
#perplex = 80
learn_rate = round((X.shape[0])/12)
#learn_rate = 400
components=2
print("perplex ",perplex)
print("learn_rate ",learn_rate)
print("components ",components)
m = TSNE(n_components=components,learning_rate=learn_rate,perplexity=perplex, random_state=1, init="pca").fit_transform(X)
print(m.shape)

plt.rcParams['font.size'] = 15
target_ids = [0,1]
colors = ["r","b"]
#zettel = np.concatenate((np.full((sample_n),"linker"),np.full((sample_n),"domain")))
colors = np.concatenate((np.full((sample_n),"r"),np.full((sample_n),"b")))
Y = Y.reset_index(drop=True)
S = Y[Y[1026]==0]
Sm = m[S.index]
W = Y[Y[1026]==1]
Wm = m[W.index]
T = Y[Y[1026]==2]
Tm = m[T.index]
colors_S = np.full((S.index.shape[0]), "g")
colors_W = np.full((W.index.shape[0]), "b")
colors_T = np.full((T.index.shape[0]), "r")
colors_nP = np.full((m[sample_n:,1].shape[0]),"b")
colors_pP = np.full((sample_n),"r")

m_df = pd.DataFrame(m)


fig, ax = plt.subplots()
#ax.scatter(m[sample_n:,0], m[sample_n:,1], c=colors_nP, label="nP", alpha=0.4, s=50)
#ax.scatter(m[:sample_n,0], m[:sample_n,1], c=colors_pP, label="pP", alpha=0.4, s=50)

ax.scatter(Sm[:,0], Sm[:,1], c=colors_S, label="S", alpha=0.4, s=50)
ax.scatter(Wm[:,0], Wm[:,1], c=colors_W, label="Y", alpha=0.4, s=50)
ax.scatter(Tm[:,0], Tm[:,1], c=colors_T, label="T", alpha=0.4, s=50)


fig.suptitle("samples: " + str(m.shape[0]) + "\nperplexity: " + str(perplex) + "\nlearning rate: " + str(learn_rate) + "\nn_components: " + str(components))
ax.legend()
fig.show()
print("donw")
