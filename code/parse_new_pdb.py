import math
from pathlib import Path
from Bio.PDB import *
io = PDBIO()
# PDB parser
parser = PDBParser()
# SeqIO parser (getting sequences)
from Bio import SeqIO
import os




WD = Path(__file__).resolve().parents[1]

def remove_except_CA(structure, true_chain):
    for model in structure:
        clist = list(model.child_dict.keys())
        clist.remove(true_chain)
        for i in clist:
            if len(model.child_dict.keys()) > 1:
                model.detach_child(i)
        for chain in model:
            # supposed to remove residues that were added to aid in expression = resnumber < 1 but disabled because phosphate location in there
            # res_keys = list(chain.child_dict.keys())
            # for res in res_keys:
            #     if res[1] <= 0:
            #         chain.detach_child(res)
            for residue in chain:
                keys = list(residue.child_dict.keys())
                for atom in keys:
                    if atom != "CA":
                        residue.detach_child(atom)
    return structure



import pickle as pkl

# add pdb files to the pdb_collection after removing the phosphate group and the other chains
# SEP (S)	phosphoserine
# TPO (T)	phosphothreonine
# PTR (Y)	O-phosphotyrosine



import warnings
warnings.filterwarnings("ignore")
def modifypdb(file, main_chain, modelid):
    structure = parser.get_structure(file[:-4], (WD/"ML_data"/"new_pdbs"/file))
    phospho_pos = set()
    structure = remove_except_CA(structure, main_chain)
    chain = structure.child_dict[modelid].child_dict[main_chain]
    for residue in chain:
        #                    for atom in residue:
        #                        if atom.element=="H":
        #                            residue.detach_child(atom.id)
        if "P" in residue and (residue.resname == "SER" or residue.resname == "TYR" or residue.resname == "THR"):
            print(str(structure.id) + " hit at " + str(residue.id[1]) + " model " + str(modelid))
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
            print(str(structure.id)+" hit at "+ str(residue.id[1]) + " model " + str(modelid))
    clist = list(structure.child_dict.keys())
    clist.remove(modelid)
    for i in clist:
        if len(structure.child_dict.keys()) > 1:
            structure.detach_child(i)

    pable_residues = []
    for residue in structure.get_residues():
        if residue.id[2]==" ":
            if residue.resname == "SER" or residue.resname == "TYR" or residue.resname == "THR":
                if float(residue.id[1]) not in pable_residues:
                    #print("stop")
                    pable_residues.append(float(residue.id[1]))

    linewrite_p_able = ""
    for pa in pable_residues:
        linewrite_p_able += "," + str(pa)
    linewrite_p_able = linewrite_p_able[1:]

    sequence_start = structure.child_list[0].child_list[0].child_list[0].id[1]
    sequence_end = structure.child_list[0].child_list[0].child_list[-1].id[1]
    iop_file = WD / "phos_info" / "info_on_new_pdb_phos.txt"
    with open(iop_file, 'a') as iop:
        iopfile = iop

        iopfile.write((structure.id+ "\t" + main_chain + "\t" + str(sequence_start) + "\t" + str(sequence_end) + "\t" + str(linewrite_p_able) + "\n"))
    io.set_structure(structure)
    io.save(str(WD/"ML_data"/"parsed_new_pdbs"/(structure.id+".txt")), preserve_atom_numbering = True)

# removes all phosphates from the atoms
clean_file = WD / "phos_info" / "info_on_new_pdb_phos.txt"
iop = open(clean_file, 'w')

modifypdb("7TVS.pdb", "A", 0)
print("done")


print(WD)


