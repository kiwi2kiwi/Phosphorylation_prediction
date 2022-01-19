import numpy as np
import matplotlib.pyplot as plt


class AtomClass:
    def __init__(self, index, atom, element, x, y, z, Aminoacid, residue_number):
        super(AtomClass, self).__init__()
        self.index = index
        self.atom = atom
        self.element = element
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.Aminoacid = Aminoacid
        self.residue_number = residue_number

atom_list = []

with open("C:\\Users\\administ\\Downloads\\pdb3tz7.ent", "r") as op:
    for line in op.readlines():
        if line[:4] =="ATOM":
#            print("stop")
            zeile = line.split()
            Atom_found = AtomClass(zeile[1], zeile[2], zeile[11], zeile[6], zeile[7], zeile[8], zeile[3], zeile[5])
            atom_list.append(Atom_found)

contact_map = np.zeros((atom_list.__len__(), atom_list.__len__()))

distance = 7

for i, ai in enumerate(atom_list):
    comp_against = atom_list#[i+1:]
    for u, ca in enumerate(comp_against):
        if abs(ai.x - ca.x) <= distance:
            if abs(ai.y - ca.y) <= distance:
                if abs(ai.z - ca.z) <= distance:
                    p1 = np.array([ai.x, ai.y, ai.z])
                    p2 = np.array([ca.x, ca.y, ca.z])
                    squared_dist = np.sum((p1 - p2) ** 2, axis=0)
                    dist = np.sqrt(squared_dist)
                    if dist <= distance:
                        contact_map[i][u] = abs(dist-distance)

fig, ax = plt.subplots()
ax.imshow(contact_map, cmap='Oranges', interpolation='nearest')
ax.invert_yaxis()
plt.show()


print("finished")


