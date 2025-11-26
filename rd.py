import os, sys, json
import numpy as np
from Bio.PDB.ResidueDepth import get_surface, residue_depth
from Bio.PDB.PDBParser import PDBParser
from utils import unzip_res_range

def calculate_rd(struct_file, resmapping, p_len):
    try:
        parser = PDBParser(QUIET=True)
        struct = parser.get_structure('PDB', struct_file)
        assert len(struct) == 1
        model = struct[0]
        assert len(model) == 1

        chains = [c for c in model]
        assert len(chains) == 1

        rd_dict = {}
        surface = get_surface(model)
        for res in chains[0]:
            res_id = res.get_id()
            rd_dict[str(res_id[1])+res_id[2].strip()] = residue_depth(res, surface)

        rd = dict([(int(resmapping[k]), rd_dict[k]) for k in rd_dict])
        rd = np.array([rd[r] if r in rd else np.nan for r in range(1, p_len+1)])
    except:
        rd = np.array([np.nan for r in range(1, p_len+1)])
    return rd


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
    rd = calculate_rd('./data/structs/'+struct_basefile, resmapping, len(fasta_dict[p]))
    with open('./data/rd/'+struct_basefile[:-4]+'.npy', 'wb') as outfile:
        np.save(outfile, rd)

print('done!')
