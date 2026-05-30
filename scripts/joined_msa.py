import os, sys
from msa import *

fasta_dict = {}
with open('./data/all_preppi_uniprot_seqs.txt', 'r') as infile:
    for line in infile:
        line_list = line.strip().split()
        fasta_dict[line_list[0]] = line_list[1]

interactions = []
with open('./data/all_preppi_interactions.txt', 'r') as infile:
    for line in infile:
        line_list = line.strip().split()
        interactions.append((line_list[0], line_list[1]))

assert len(interactions) == len(set(interactions))

for interaction in interactions:
    p1, p2 = interaction[0], interaction[1]
    cdhit_input_file1 = './data/msa/'+p1+'_rawmsa.fasta'
    cdhit_input_file2 = './data/msa/'+p2+'_rawmsa.fasta'
    cdhit_clstr_file1 = './data/msa/'+p1+'.cdhit.clstr'
    cdhit_clstr_file2 = './data/msa/'+p2+'.cdhit.clstr'
    if os.path.exists(cdhit_input_file1) and os.path.exists(cdhit_input_file2) and os.path.exists(cdhit_clstr_file1) and os.path.exists(cdhit_clstr_file2):
        joined_oneline_msa_file, joined_aligned_msa_file = calculate_joined_msa(p1, p2, fasta_dict[p1], fasta_dict[p2], cdhit_input_file1, cdhit_input_file2, cdhit_clstr_file1, cdhit_clstr_file2, './data/joined_msa')

print('done!')
