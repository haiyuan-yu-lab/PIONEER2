import os, json
import numpy as np
from tempfile import mkdtemp
from shutil import rmtree
from utils import unzip_res_range, naccess

def srescalc(struct_file, uSASA):
    scratchDir = mkdtemp()
    
    os.system('cp '+struct_file+' '+scratchDir)
    naccess_output = naccess(os.path.join(scratchDir, os.path.basename(struct_file)))
    
    output = []
    for line in naccess_output:
        if line[:3] != 'RES':
            continue

        aa = line[3:7].strip()
        residue_index = line[9:14].strip()
        relative_perc_accessible = float(line[22:28])

        if relative_perc_accessible < uSASA:
            continue
            
        output.append((residue_index, aa, relative_perc_accessible))
        
    rmtree(scratchDir)
    return output

def calculate_sasa(struct_file, resmapping, p_len, uSASA):
    output = srescalc(struct_file, uSASA)
    
    SASAs = dict([(int(resmapping[q[0]]), q[2]) for q in output])
    SASAs = np.array([SASAs[r] if r in SASAs else np.nan for r in range(1, p_len+1)])
    return SASAs


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
    sasas = calculate_sasa('./data/structs/'+struct_basefile, resmapping, len(fasta_dict[p]), -1)
    with open('./data/sasa/'+struct_basefile[:-4]+'.npy', 'wb') as outfile:
        np.save(outfile, sasas)

print('done!')
