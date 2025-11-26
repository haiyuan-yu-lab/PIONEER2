import os, pickle
import numpy as np
from protein_physical_chemistry import expasy

def calculate_expasy(seq):
    d = {}
    for prop in expasy:
        d[prop] = np.array([expasy[prop][aa] if aa in expasy[prop] else np.nan for aa in seq.upper()])
    return d


fasta_dict = {}
with open('./data/all_preppi_uniprot_seqs.txt', 'r') as infile:
    for line in infile:
        line_list = line.strip().split()
        fasta_dict[line_list[0]] = line_list[1]

with open('./data/for_sequence_extraction/updated_uniprots.txt', 'r') as infile:
    for line in infile:
        line_list = line.strip().split()
        assert line_list[0] in fasta_dict
        assert fasta_dict[line_list[0]] == line_list[1]

for identifier in sorted(list(fasta_dict.keys())):
    result = calculate_expasy(fasta_dict[identifier])
    with open('./data/expasy/'+identifier+'.pkl', 'wb') as outfile:
        pickle.dump(result, outfile)

print('done!')
