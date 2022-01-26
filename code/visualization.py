from pathlib import Path
import pickle
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import os
import pandas as pd

index_dict = {}
WD = Path(__file__).resolve().parents[1]
with open(WD / "phos_info" / "info_on_phos.txt","r") as iop:
    for line in iop.readlines():
        line = line.split()
        trues = line[2].split(",")
        trues = list(map(float, trues))
        index_dict[line[0]] = trues



emb_name = "glove"
emb_folder = os.path.join("../ML_data", "train_data", "embeddings", emb_name)
sample_n = len(os.listdir(emb_folder))
training_embs = pd.DataFrame()
for file in os.listdir(emb_folder):
    to_append = pd.DataFrame(pickle.load(open(os.path.join(emb_folder, file), "rb")))
    label_list = pd.Series(0, to_append.index)
    true_labels = index_dict[file[:-2]]
    label_list.loc[true_labels] = 1
    to_append[512] = label_list
    to_append = to_append.reset_index(drop=True)
    training_embs = training_embs.append(to_append)
training_embs = training_embs.reset_index(drop=True)
print(training_embs.shape)

false_labels = training_embs.loc[training_embs[512] == 0]
true_labels = training_embs.loc[training_embs[512] == 1]
false_labels = false_labels.sample(frac=true_labels.shape[0]/false_labels.shape[0])
sample_n = false_labels.shape[0]

perplex = round((sample_n*2)*0.01)
#perplex = 50
learn_rate = round((sample_n*2)/12)
#learn_rate = 200
components=2
m = TSNE(n_components=components,learning_rate=learn_rate,perplexity=perplex, random_state=1, init="pca").fit_transform()
print(m.shape)

plt.rcParams['font.size'] = 15
target_ids = [0,1]
colors = ["r","b"]
#zettel = np.concatenate((np.full((sample_n),"linker"),np.full((sample_n),"domain")))
colors = np.concatenate((np.full((sample_n),"r"),np.full((sample_n),"b")))
colors_linkers = np.full((sample_n),"r")
colors_domains = np.full((sample_n),"b")



fig, ax = plt.subplots()
plt=ax.scatter(tsne_linkers[:,0], tsne_linkers[:,1], c=colors_linkers, label="linker", alpha=alpha, s=size)
ax.scatter(tsne_domains[:,0], tsne_domains[:,1], c=colors_domains, label="domain", alpha=alpha, s=size)

fig.suptitle("samples: " + str(sample_n * 2) + "\nperplexity: " + str(perplex) + "\nlearning rate: " + str(learn_rate) + "\nn_components: " + str(components))
ax.legend()
fig.show()
