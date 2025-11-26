import os, pickle
import pandas as pd
import numpy as np

def calculate_pssm(pssm_file, w=11):
    with open(pssm_file, 'r') as infile:
        line = infile.readline()
        line = infile.readline()
        line = infile.readline()
        residue_cols = line.strip().split()[:20]
        
        seq = ''
        pssm_ori = []
        for line in infile:
            if line == '\n':
                break
                
            line_list = line.strip().split()
            seq += line_list[1]
            pssm_ori.append([float(i) for i in line_list[2:22]])
            
        pssm_ori = pd.DataFrame(np.array(pssm_ori), columns=residue_cols)
        
        pssm_window = []
        for i in range(pssm_ori.shape[0]):
            if i-(w-1)/2 < 0:
                if i+(w-1)/2 < pssm_ori.shape[0]:
                    pssm_window.append(np.sum(pssm_ori[0:int(i+(w-1)/2+1)], axis=0))
                else:
                    pssm_window.append(np.sum(pssm_ori[0:pssm_ori.shape[0]], axis=0))
            else:
                if i+(w-1)/2 < pssm_ori.shape[0]:
                    pssm_window.append(np.sum(pssm_ori[int(i-(w-1)/2):int(i+(w-1)/2+1)], axis=0))
                else:
                    pssm_window.append(np.sum(pssm_ori[int(i-(w-1)/2):pssm_ori.shape[0]], axis=0))
                    
        pssm_window = pd.DataFrame(np.array(pssm_window), columns=residue_cols)
    return seq, pssm_ori, pssm_window


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
    pssm_file = './data/psiblast_uniref90/'+identifier+'.pssm'
    if os.path.exists(pssm_file):
        seq, pssm_ori, pssm_window = calculate_pssm(pssm_file, 11)
        assert seq == fasta_dict[identifier]
        pssm_ori.to_pickle('./data/pssm_uniref90/'+identifier+'_ori.pkl')
        pssm_window.to_pickle('./data/pssm_uniref90/'+identifier+'_window.pkl')
    else:
        pass

print('done!')
    