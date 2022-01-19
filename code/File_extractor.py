import math
from pathlib import Path

class Structure:
    def __init__(self, name, phosphorylated_positions, phosphorylated):
        self.phosphorylated=[]
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

    def add_protein_structure(self, Structure):
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
            pP = line[1]
            wP = line[6]
            id = line[4]
            pP_Structure = Structure(pP, line[3], True)
            pP_Structure.add_protein_id(id)
            wP_Structure = Structure(wP, line[8], False)
            wP_Structure.add_protein_id(id)
            if id in Protein_list:
                Protein_list[id].add_protein_structure(pP_Structure)
                Protein_list[id].add_protein_structure(wP_Structure)
            else:
                new_Protein = Protein(id)
                new_Protein.add_protein_structure(pP_Structure)
                new_Protein.add_protein_structure(wP_Structure)
                Protein_list[id] = new_Protein

structure_id_association(ser90_paired)
structure_id_association(thr90_paired)
structure_id_association(tyr90_paired)


Protein_structure_dict={}
# makes the protein id searchable by structure name
for name, Protein in Protein_list.items():
    for structu_name, structu in Protein.protein_structures.items():
        Protein_structure_dict[structu_name]=name


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


from shutil import copy2
import math
def copy_least_phosphorylated_to_pdb_collection():
    iop_file = WD / "phos_info" / "info_on_phos.txt"
    with open(iop_file, 'w') as iop:
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
            print(dst)
            if Path(best_file).name == "not set":
                print("stop")
            copy2(best_file, dst)
            linewrite_p = ""
            for p in best_s.phosphorylated:
                linewrite_p += "," + str(float(p))
            linewrite_pos = ""
            for pos in best_s.phosphorylated_positions:
                linewrite_pos += "," + str(pos)
            linewrite_p = linewrite_p[1:]
            linewrite_pos = linewrite_pos[1:]

            # this file consists of the structure name, if residues are phosphorylated in this structure = 1, or where we know residues are phosphorylated in other structures = 0, and their positions
            iop.write((best_s.name+ "\t"+ linewrite_p+ "\t"+ linewrite_pos + "\n"))
    iop.close()


copy_least_phosphorylated_to_pdb_collection()
import pickle as pkl
pkl.dump(protein_files_of_wp, open((WD / "phos_info" / "structure_locations.txt"), "wb"))

print(WD)


