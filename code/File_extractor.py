import math
from pathlib import Path
from Bio.PDB import *
io = PDBIO()
# PDB parser
parser = PDBParser()
# SeqIO parser (getting sequences)
from Bio import SeqIO
import os


class Structure:
    def __init__(self, name, phosphorylated_positions, phosphorylated, chain):
        self.phosphorylated=[]
        self.chain = chain
        self.phosphorylated.append(bool(phosphorylated))
        self.name = name
        self.phosphorylated_positions=[]
        self.phosphorylated_positions.append(float(phosphorylated_positions))
        self.count_nonP_sites = 0
        self.count_pP_sites = 0
        self.location = "not set"
        if phosphorylated:
            self.count_pP_sites += 1
        else:
            self.count_nonP_sites += 1

    def add_new_position(self, phos, phos_pos):
        if self.name =="3k2l":
            print("stop")
        self.phosphorylated.append(bool(phos))
        self.phosphorylated_positions.append(float(phos_pos))
        if phos:
            self.count_pP_sites += 1
        else:
            self.count_nonP_sites += 1
        print("multiple phosphorylations for Structure ", self.name, " in Protein ", self.id)

    def add_protein_id(self, id):
        self.id = id

    def add_file_location(self, location):
        self.location = location

class Protein:
    def __init__(self, id):
        self.protein_structures = {}
        self.id = id

    def add_protein_structure(self, Structure, chain):
        if Structure.name in self.protein_structures:
            self.protein_structures[Structure.name].add_new_position(bool(Structure.phosphorylated), Structure.phosphorylated_positions[0])#phosphorylated.append(Structure.phosphorylated)
#            self.protein_structures[Structure.name].phosphorylated_positions.append(Structure.phosphorylated_positions)
        else:
            self.protein_structures[Structure.name]=Structure
            if len(self.protein_structures) == 2:
                print("two structures for protein ",self.id)
            elif len(self.protein_structures) > 2:
                print("multiple structures for protein ", self.id)

WD = Path(__file__).resolve().parents[1]

ser90 = WD / 'data' / "ser90" / "wP" / "wP_paired" / "list_wP.txt"
thr90 = WD / 'data' / "thr90" / "wP" / "wP_paired" / "list_wP.txt"
tyr90 = WD / 'data' / "tyr90" / "wP" / "wP_paired" / "list_wP.txt"
# list_wp contains the positions of the phosphorylation sites coupled to the structure name

ser90_paired = WD / 'data' / "ser90" / "wP" / "wP_paired" / "list_paired_wP_pP.txt"
thr90_paired = WD / 'data' / "thr90" / "wP" / "wP_paired" / "list_paired_wP_pP.txt"
tyr90_paired = WD / 'data' / "tyr90" / "wP" / "wP_paired" / "list_paired_wP_pP.txt"
# list_paired_wP_pP contains the structures of pP and wP with protein id

ser90_prot_files = WD / 'data' / "ser90" / "wP" / "wP_paired" / "dali"
thr90_prot_files = WD / 'data' / "thr90" / "wP" / "wP_paired" / "dali"
tyr90_prot_files = WD / 'data' / "tyr90" / "wP" / "wP_paired" / "dali"
# files in the dali directory don't contain the protein id, only structure name and structure

Protein_list={}
# creating a dictionary for the protein identifiers from the paired files
def structure_id_association(file):
    with open(file) as f:
        for line in f.readlines():
            line = line.split()
            pchain = line[2]
            wchain = line[7]
            pP = line[1]
            wP = line[6]
            id = line[4]
            #if chain == "K" and id == "6b5b":
             #   print("stop")
            pP_Structure = Structure(pP, line[3], True,pchain)
            pP_Structure.add_protein_id(id)
            wP_Structure = Structure(wP, line[8], False,wchain)
            wP_Structure.add_protein_id(id)
            if id in Protein_list:
                Protein_list[id].add_protein_structure(pP_Structure,pchain)
                Protein_list[id].add_protein_structure(wP_Structure,wchain)
            else:
                new_Protein = Protein(id)
                new_Protein.add_protein_structure(pP_Structure,pchain)
                new_Protein.add_protein_structure(wP_Structure,wchain)
                Protein_list[id] = new_Protein

structure_id_association(ser90_paired)
structure_id_association(thr90_paired)
structure_id_association(tyr90_paired)


Protein_structure_dict={}
# makes the protein id searchable by structure name
for name, Protein in Protein_list.items():
    for structu_name, structu in Protein.protein_structures.items():
        Protein_structure_dict[structu_name]=name

def remove_except_CA(structure, true_chain):
    for model in structure:
        clist = list(model.child_dict.keys())
        clist.remove(true_chain)
        for i in clist:
            if len(model.child_dict.keys()) > 1:
                model.detach_child(i)
#        for chain in model:
#            if chain.id != true_chain:
#                model.detach_child(chain.id)
        for chain in model:
            res_keys = list(chain.child_dict.keys())
            for res in res_keys:
                if res[1] <= 0:
                    chain.detach_child(res)
            for residue in chain:
                keys = list(residue.child_dict.keys())
                for atom in keys:
                    if atom != "CA":
                        residue.detach_child(atom)
    return structure

# extracts information from the list_wp.txt file
"3kvw	A	159.0	28.635	-33.739	7.191"
def extract_list_wp(file):
    with open(file) as f:
        for line in f.readlines():
            line = line.split()
            wP = line[0]
            if id in Protein_structure_dict:
                id = Protein_structure_dict[wP]
                existing_Structure = Protein_list[id].protein_structures[wP]
                if existing_Structure.phosphorylated_positions.contains(float(line[2])):
                    print("double detection of wP site in list_wp")
                else:
                    existing_Structure.add_new_position(False, float(line[2]))
            else:
                print("new structure in list_wp file")


protein_files_of_wp={}
# creates a dictionary of the protein structure with its location
def create_dictionary_of_protein_files(p):
    for child in p.iterdir():
        if child.stem == '1u9i':
            print("stop")
        structure_name = child.stem
        structure_name.split("_")
        protein_files_of_wp[structure_name] = str(child)
        print(str(structure_name), "\t", str(child))


create_dictionary_of_protein_files(ser90_prot_files)
create_dictionary_of_protein_files(thr90_prot_files)
create_dictionary_of_protein_files(tyr90_prot_files)

def path_to_structure():
    for key, val in protein_files_of_wp.items():
        if key in Protein_structure_dict:
            Prot_id = Protein_structure_dict[key]
            Protein_list[Prot_id].protein_structures[key].add_file_location(val)
    # now check if every structure has its blueprint
    for key_prot, Prot in Protein_list.items():
        for key_stru, Stru in Prot.protein_structures.items():
            if Stru.location == "not set" and len(Stru.phosphorylated) != sum(Stru.phosphorylated):
                print("Protein ", key_prot, " Structure ", key_stru, " has no pdb structure blueprint")

path_to_structure()


import math
iopfile = ""
def copy_least_phosphorylated_to_pdb_collection():
    iop_file = WD / "phos_info" / "info_on_phos.txt"
    with open(iop_file, 'w') as iop:
        iopfile = iop
        for i in Protein_list.values():
            best_count = 0
            best_file = "none"
            best_s = "none"
            for s in i.protein_structures.values():
                if s.id == 'Q79PF4':
                    print("stop")
                watch = len(s.phosphorylated) - sum(s.phosphorylated)
                if (len(s.phosphorylated) - sum(s.phosphorylated)) >= best_count:
                    best_file = s.location
                    best_count = len(s.phosphorylated) - sum(s.phosphorylated)
                    best_s = s
            dst = WD / 'pdb_collection' / Path(best_file).name
            if Path(best_file).name == "not set":
                print("stop")

            strt = parser.get_structure(best_s.name, best_s.location)
            sequence_start = strt.child_list[0].child_list[0].child_list[0].id[1]
            sequence_end = strt.child_list[0].child_list[0].child_list[-1].id[1]
            strt = remove_except_CA(strt,best_s.chain)
            io.set_structure(strt)
            io.save(str(WD / "pdb_collection" / (strt.id + ".txt")), preserve_atom_numbering=True)

            print(dst)

#            copy2(best_file, dst)
            linewrite_p = ""
            for p in best_s.phosphorylated:
                linewrite_p += "," + str(float(p))
            linewrite_pos = ""
            for pos in best_s.phosphorylated_positions:
                linewrite_pos += "," + str(pos)
            linewrite_p = linewrite_p[1:]
            linewrite_pos = linewrite_pos[1:]

            # this file consists of the structure name, if residues are phosphorylated in this structure = 1, or where we know residues are phosphorylated in other structures = 0, and their positions
            iop.write((best_s.name + "\t" + linewrite_p + "\t" + linewrite_pos + "\t" + best_s.chain  + "\t" + str(sequence_start) + "\t" + str(sequence_end) + "\n"))


copy_least_phosphorylated_to_pdb_collection()
import pickle as pkl
pkl.dump(protein_files_of_wp, open((WD / "phos_info" / "structure_locations.txt"), "wb"))

# add pdb files to the pdb_collection after removing the phosphate group and the other chains
# SEP (S)	phosphoserine
# TPO (T)	phosphothreonine
# PTR (Y)	O-phosphotyrosine
# remo


pStruc_dict={}
class pStruc:
    def __init__(self, name, chain, position):
        self.name = name
        self.chain = chain
        self.position=[]
        self.position.append(position)


ser90_pP = WD / 'data' / "ser90" / "pP" / "list_pP.txt"
thr90_pP = WD / 'data' / "thr90" / "pP" / "list_pP.txt"
tyr90_pP = WD / 'data' / "tyr90" / "pP" / "list_pP.txt"

def download_pdbs(file_path):
    with open(file_path) as file:
        for line in file.readlines():
            spltted = line.split()
            id = spltted[0]
            if id not in pStruc_dict:
                pStruc_dict[id]=pStruc(id,spltted[1],spltted[2])
            else:
                pStruc_dict[id].position.append(spltted[2])
download_pdbs(ser90_pP)
download_pdbs(thr90_pP)
download_pdbs(tyr90_pP)

from Bio.PDB import PDBList
pdbl = PDBList()
downloaded=[]
for file in os.listdir((WD/"pdb_collection_pP")):
#    print(file)
    downloaded.append(file[3:-4])
downloaded = set(downloaded)
to_download=set(pStruc_dict.keys()).difference(downloaded)

pdbl.download_pdb_files(pdb_codes=to_download, obsolete=False, pdir=(WD/"pdb_collection_pP"), file_format="pdb", overwrite=False)

import warnings
warnings.filterwarnings("ignore")
def modifypdbs(pdb_dir):
    for file in os.listdir(pdb_dir):
        structure = parser.get_structure(file[3:-4], (WD/"pdb_collection_pP"/file))
        phospho_pos = set()
        structure = remove_except_CA(structure, pStruc_dict[structure.id].chain)
        model_max_dict = {}
        for model in structure:
            model_max_dict[model.id] = 0
            for chain in model:
                for residue in chain:
#                    for atom in residue:
#                        if atom.element=="H":
#                            residue.detach_child(atom.id)
                    if "P" in residue and (residue.resname == "SER" or residue.resname == "TYR" or residue.resname == "THR"):
                        phospho_pos.add(residue.id[1])
                        model_max_dict[model.id] = model_max_dict[model.id]+1
                        print(str(structure.id) + " hit at " + str(residue.id[1]) + " model " + str(model.id))
                    if (residue.resname == "SEP" or residue.resname == "TPO" or residue.resname == "PTR"):
                        # commented because they get removed anyway
                        #try:
                        #    residue.detach_child("P")
                        #except:
                        #    print("haha funny exception")
                        #try:
                        #    residue.detach_child("O1P")
                        #except:
                        #    print("haha funny exception")
                        #try:
                        #    residue.detach_child("O2P")
                        #except:
                        #    print("haha funny exception")
                        #try:
                        #    residue.detach_child("O3P")
                        #except:
                        #    print("haha funny exception")
                        if residue.resname == "SEP":
                            residue.resname = "SER"
                            resid = list(residue.id)
                            resid[0] =""
                            residue.id = tuple(resid)
                        if residue.resname == "TPO":
                            residue.resname = "THR"
                            resid = list(residue.id)
                            resid[0] =""
                            residue.id = tuple(resid)
                        if residue.resname == "PTR":
                            residue.resname = "TYR"
                            resid = list(residue.id)
                            resid[0] =""
                            residue.id = tuple(resid)
                        phospho_pos.add(residue.id[1])
                        model_max_dict[model.id] = model_max_dict[model.id] + 1
                        print(str(structure.id)+" hit at "+ str(residue.id[1]) + " model " + str(model.id))
        print(model_max_dict)
        maxhit = max(model_max_dict, key=model_max_dict.get)
        clist = list(structure.child_dict.keys())
        clist.remove(maxhit)
        for i in clist:
            if len(structure.child_dict.keys()) > 1:
                structure.detach_child(i)

        linewrite_pos = ""
        for pos in phospho_pos:
            linewrite_pos += "," + str(pos)
        linewrite_p = ""
        for p in phospho_pos:
            linewrite_p += ",1"
        linewrite_p = linewrite_p[1:]
        linewrite_pos = linewrite_pos[1:]

        sequence_start = structure.child_list[0].child_list[0].child_list[0].id[1]
        sequence_end = structure.child_list[0].child_list[0].child_list[-1].id[1]
        iop_file = WD / "phos_info" / "info_on_phos.txt"
        with open(iop_file, 'a') as iop:
            iopfile = iop

            iopfile.write((structure.id+ "\t"+ linewrite_p+ "\t"+ linewrite_pos + "\t" + pStruc_dict[structure.id].chain + "\t" + str(sequence_start) + "\t" + str(sequence_end) + "\n"))
        io.set_structure(structure)
        io.save(str(WD/"pdb_collection"/(structure.id+".txt")), preserve_atom_numbering = True)

# removes all phosphates from the atoms
modifypdbs(WD/"pdb_collection_pP")



print(WD)


