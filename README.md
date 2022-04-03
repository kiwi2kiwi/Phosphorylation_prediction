# Phosphorylation_prediction
Bioinformatics research project at Charles University. Supervisors are Marian Novotny and David Hoksza with help from Hamza Gamouh and Michael Heinzinger.
I'm trying to predict phosphorylated residues from sequence and structure data
Prediction algorithm is a GNN

How do i use this?

For Training:
from the file 5dlt, the last residue has to be removed:\
HETATM 6575 CA    CA A 907\
otherwise it will create a conflict with the sequence length

For Prediction:\
1. install the anaconda environment "environment.yml"\
2. Download a protein structure file. It has to be in the .pdb format. Put it in the ML_data/new_pdbs directory.\
3. Head to line 129 of the parse_new_pdb.py file. Change the file name to your protein file name. Provide model and chain id.\
4. Run emb_graph_new_pdb.py to generate embeddings and a graph
5. Run predictor.py to obtain the predictions in a format that is usable in PyMol

