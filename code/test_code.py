#import matplotlib
#import matplotlib.pyplot as plt
#matplotlib.use("TkAgg")
#plt.scatter(x=[1,2,3], y = [3,2,1])
#plt.show()


from pathlib import Path
import math
from Bio.PDB import *
io = PDBIO()
# PDB parser
parser = PDBParser()
# SeqIO parser (getting sequences)
from Bio import SeqIO
import os
WD = Path(__file__).resolve().parents[1]




import numpy as np
from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score, confusion_matrix
from sklearn.metrics import matthews_corrcoef as MCC
print(MCC(np.array([1,1,1,2,1,1]), np.array([1,1,1,1,1,2])))
print(accuracy_score(np.array([1,1,1,2,1,1]), np.array([1,1,1,1,1,2])))
print(f1_score(np.array([1,1,1,2,1,1]), np.array([1,1,1,1,1,2])))
cm = confusion_matrix(np.array([1,1,1,2,1,1]), np.array([1,1,1,1,1,2]))
cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]



import bio_embeddings.embed.prottrans_albert_bfd_embedder as prottrans_albert_bfd_embedder
Embedder = prottrans_albert_bfd_embedder.ProtTransAlbertBFDEmbedder()
embedded = Embedder.embed("MLSD")




from EarlyStopping import EarlyStopping
early_stopping = EarlyStopping(patience=1, path="..\\ML_data\\model.pt")
early_stopping(0.5,[1])
early_stopping(0.4,[2])
early_stopping(0.45,[3])



#global var
var = 4

def foo1():
    global var
    var += 1

foo1()
foo1()
print(var)



for file in os.listdir((WD/"demo")):
    structure = parser.get_structure(file[:-4], (WD / "demo" / file))
    phospho_pos = set()
    for model in structure:
        for chain in model:
            for residue in chain:
                if "P" in residue or (
                        residue.resname == "SEP" or residue.resname == "PTO" or residue.resname == "PTR"):
                    try:
                        residue.detach_child("P")
                    except:
                        print("haha funny exception")
                    try:
                        residue.detach_child("O1P")
                    except:
                        print("haha funny exception")
                    try:
                        residue.detach_child("O2P")
                    except:
                        print("haha funny exception")
                    try:
                        residue.detach_child("O3P")
                    except:
                        print("haha funny exception")
                    if residue.resname == "SEP":
                        residue.resname = "SER"
                        resid = list(residue.id)
                        resid[0] = ""
                        residue.id = tuple(resid)
                    if residue.resname == "TPO":
                        residue.resname = "THR"
                        resid = list(residue.id)
                        resid[0] = ""
                        residue.id = tuple(resid)
                    if residue.resname == "PTR":
                        residue.resname = "TYR"
                        resid = list(residue.id)
                        resid[0] = ""
                        residue.id = tuple(resid)
                    phospho_pos.add(residue.id[1])
        break
#io.set_structure(structure)
#io.save(str(WD/"demo"/(structure.id+"_changed.txt")), preserve_atom_numbering = True)













lst = []
with open((WD/"pdb_collection"/"4fbn.txt"),"r") as r:
    for ln in r.readlines():
        lst.append(float(ln.split()[5]))

print(list(set(lst)).__len__())

lst = list(set(lst))
#lst = [1,2,3,4,5,7,8]
counter = 1
for i in lst[1:]:
    if i - lst[counter-1] != 1:
        print(i)
    counter += 1

[1,2,3].__contains__([1,2])

print("end")