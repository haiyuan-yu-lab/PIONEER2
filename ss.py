import os, json
import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import dssp_dict_from_pdb_file
from utils import unzip_res_range

def ss8to3(ss):
    if ss == 'H' or ss == 'G' or ss == 'I':
        return [1, 0, 0] #'H'
    elif ss == 'B' or ss == 'E':
        return [0, 1, 0] #'E'
    else:
        assert ss == 'T' or ss == 'S' or ss == '-'
        return [0, 0, 1] #'C'
    
def calculate_ss(struct_file, resmapping, p_len):
    parser = PDBParser(QUIET=True)

    struct = parser.get_structure('PDB', struct_file)
    assert len(struct) == 1
    model = struct[0]
    assert len(model) == 1

    chains = [c for c in model]
    assert len(chains) == 1

    dssp_tuple = dssp_dict_from_pdb_file(struct_file, DSSP='mkdssp')
    dssp_dict = dssp_tuple[0]

    ss = dict([(int(resmapping[str(k[1][1])+k[1][2].strip()]), dssp_dict[k][1]) for k in dssp_dict])
    ss = np.array([ss8to3(ss[r]) if r in ss else [np.nan, np.nan, np.nan] for r in range(1, p_len+1)])
    return ss


fasta_dict = {}
with open('./data/all_preppi_uniprot_seqs.txt', 'r') as infile:
    for line in infile:
        line_list = line.strip().split()
        fasta_dict[line_list[0]] = line_list[1]

with open('./data/structs_to_use.json', 'r') as infile:
    structures = json.load(infile)

all_interactions = sorted(list(structures.keys()))

all_structs = []
for pp in all_interactions:
    p1, p2 = pp.split('_')
    assert p1 <= p2

    p1_structs = structures[p1+'_'+p2][p1]
    for i, p1_struct in enumerate(p1_structs):
        if (p1, p1_struct) not in all_structs:
            all_structs.append((p1, p1_struct))

    if p1 != p2:
        p2_structs = structures[p1+'_'+p2][p2]
        for i, p2_struct in enumerate(p2_structs):
            if (p2, p2_struct) not in all_structs:
                all_structs.append((p2, p2_struct))

for struct in sorted(all_structs):
    p, p_struct = struct[0], struct[1]

    if p_struct[0] == 'PDB':
        assert p_struct[4] != ' '
        struct_basefile = p+'_'+p_struct[3]+'_'+p_struct[4]+'.pdb'
    else:
        assert p_struct[0] == 'AlphaFold'
        struct_basefile = p+'_AF.pdb'

    uniprot_range, pdb_range = p_struct[1], p_struct[2]
    resmapping = dict(zip(unzip_res_range(pdb_range), unzip_res_range(uniprot_range)))
    ss = calculate_ss('./data/structs/'+struct_basefile, resmapping, len(fasta_dict[p]))
    with open('./data/ss/'+struct_basefile[:-4]+'.npy', 'wb') as outfile:
        np.save(outfile, ss)

print('done!')
