# Phosphorylation_prediction
Bioinformatics research project at Charles University. Supervisors are Marian Novotny and David Hoksza with help from Hamza Gamouh and Michael Heinzinger
I'm trying to predict phosphorylated residues from sequence and structure data
Prediction algorithm is a GNN

The pdb_collection contains a folder with pdb entries, that have been reduced to their CA atoms. 
This is done to speed up the contact map



from the file 5dlt, the last residue has to be removed:
HETATM 6575 CA    CA A 907

Layer size is amount of filters
hops is kernel size