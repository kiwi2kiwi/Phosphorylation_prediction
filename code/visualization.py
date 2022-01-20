from pathlib import Path
import pickle
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import os
import pandas as pd

emb_name = "glove"
emb_folder = os.path.join("../ML_data", "train_data", "embeddings", emb_name)
sample_n = len(os.listdir(emb_folder))
training_embs = pd.DataFrame()
for file in os.listdir(emb_folder):
    training_embs = training_embs.append(pd.DataFrame(pickle.load(open(os.path.join(emb_folder, file), "rb"))))
print(training_embs.shape)

start_dict = {}
WD = Path(__file__).resolve().parents[1]
with open(WD / "phos_info" / "info_on_phos.txt","r") as iop:
    for line in iop.readlines():
        line = line.split()


perplex = round((sample_n*2)*0.01)
perplex = 50
learn_rate = 200
learn_rate = round((sample_n*2)/12)
components=2
m = TSNE(n_components=components,learning_rate=learn_rate,perplexity=perplex, random_state=1, init="pca").fit_transform()
print(m.shape)



fig, ax = plt.subplots()
plt=ax.scatter(tsne_linkers[:,0], tsne_linkers[:,1], c=colors_linkers, label="linker", alpha=alpha, s=size)
ax.scatter(tsne_domains[:,0], tsne_domains[:,1], c=colors_domains, label="domain", alpha=alpha, s=size)

fig.suptitle("samples: " + str(sample_n * 2) + "\nperplexity: " + str(perplex) + "\nlearning rate: " + str(learn_rate) + "\nn_components: " + str(components))
ax.legend()
fig.show()
