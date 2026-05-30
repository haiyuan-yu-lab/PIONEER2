import os, sys, json, pickle
from tempfile import mkdtemp
from shutil import rmtree
from collections import defaultdict
import numpy as np
from utils import unzip_res_range, naccess

def to_complex(struct_file1, struct_file2, complex_file):
    with open(struct_file1, 'r') as infile1:
        lines1 = infile1.readlines()
        
    with open(struct_file2, 'r') as infile2:
        lines2 = infile2.readlines()
        
    with open(complex_file, 'w') as outfile:
        for line in lines1:
            assert line.startswith('ATOM')
            outfile.write(line[:21]+'A'+line[22:])
        for line in lines2:
            outfile.write(line[:21]+'B'+line[22:])

def calc_ires(pdb_file, chain1, chain2, p1, p2, resmapping1, resmapping2, p1_len, p2_len, uSASA=15.0, dSASA=1.0):
    pdbatomdict = defaultdict(list)
    with open(pdb_file, 'r') as infile:
        for line in infile:
            assert 'ENDMDL' not in line and 'TER' not in line
            if line.startswith('ATOM'):
                if line[21] in [chain1, chain2]:
                    pdbatomdict[line[21]].append(line)

    if chain1 not in pdbatomdict or chain2 not in pdbatomdict:
        sys.exit('One or both chains %s and %s not found in file' %(chain1, chain2))
        
    scratchDir = mkdtemp()
    asadict = defaultdict(dict)
    for chain in pdbatomdict:
        if chain == ' ':
            tmp_pdb_file = os.path.join(scratchDir, '_.pdb')
        else:
            tmp_pdb_file = os.path.join(scratchDir, chain+'.pdb')
            
        with open(tmp_pdb_file,'w') as tmpoutfile:
            for line in pdbatomdict[chain]:
                tmpoutfile.write(line)

        naccess_output = naccess(tmp_pdb_file)
        for line in naccess_output:
            if line[:3] != 'RES':
                continue
                
            res, chain, res_num = line[3:7].strip(), line[7:9].strip(), line[9:14].strip()
            all_atoms_abs, all_atoms_rel = float(line[14:22].strip()), float(line[22:28].strip())
            if all_atoms_rel >= uSASA:
                residue_key = (res, chain, res_num)
                asadict[chain][residue_key] = all_atoms_abs
                
    if chain1 == ' ' and chain2 != ' ':
        tmp_pdb_file = os.path.join(scratchDir, '__'+chain2+'.pdb')
    elif chain1 != ' ' and chain2 == ' ':
        tmp_pdb_file = os.path.join(scratchDir, chain1+'__'+'.pdb')
    elif chain1 == ' ' and chain2 == ' ':
        tmp_pdb_file = os.path.join(scratchDir, '___'+'.pdb')
    else:
        tmp_pdb_file = os.path.join(scratchDir, chain1+'_'+chain2+'.pdb')
        
    with open(tmp_pdb_file, 'w') as tmpoutfile:
        for line in pdbatomdict[chain1]:
            tmpoutfile.write(line)
        for line in pdbatomdict[chain2]:
            tmpoutfile.write(line)

    naccess_output = naccess(tmp_pdb_file)

    intreslist = defaultdict(list)
    for line in naccess_output:
        if line[:3] != 'RES':
            continue
            
        res, chain, res_num = line[3:7].strip(), line[7:9].strip(), line[9:14].strip()
        residue_key = (res, chain, res_num)
        if residue_key not in asadict[chain]:
            intreslist[chain].append((chain, res_num, res, 0))
            continue
        
        all_atoms_abs = float(line[14:22].strip())
        res_dSASA = abs(asadict[chain][residue_key] - all_atoms_abs)
        
        if res_dSASA >= dSASA:
            intreslist[chain].append((chain, res_num, res, res_dSASA, 1))
        else:
            intreslist[chain].append((chain, res_num, res, res_dSASA, 0))
            
    ires1 = dict([(int(resmapping1[q[1]]), q[-1]) for q in intreslist[chain1]])
    ires1 = [ires1[r] if r in ires1 else np.nan for r in range(1, p1_len+1)]
    ires2 = dict([(int(resmapping2[q[1]]), q[-1]) for q in intreslist[chain2]])
    ires2 = [ires2[r] if r in ires2 else np.nan for r in range(1, p2_len+1)]
    
    if p1 == p2:
        ires_ = []
        for r in range(p1_len):
            if ires1[r] == 1 or ires2[r] == 1:
                ires_.append(1)
            else:
                assert (ires1[r] == 0 and ires2[r] == 0) or (np.isnan(ires1[r]) and np.isnan(ires2[r]))
                ires_.append(ires1[r])
        ires1 = ires_
        ires2 = ires_
        
    rmtree(scratchDir)
    return [np.array(ires1), np.array(ires2)]


fasta_dict = {}
with open('./data/all_preppi_uniprot_seqs.txt', 'r') as infile:
    for line in infile:
        line_list = line.strip().split()
        fasta_dict[line_list[0]] = line_list[1]

with open('./data/structs_to_use.json', 'r') as infile:
    structures = json.load(infile)

all_interactions = sorted(list(structures.keys()))
for interaction in all_interactions:
    p1, p2 = interaction.split('_')
    assert p1 <= p2

    if p1 == p2:
        p1_structs = structures[p1+'_'+p2][p1]
        for i, p1_struct in enumerate(p1_structs):
            if p1_struct[0] == 'PDB':
                assert p1_struct[4] != ' '
                pdb, chain = p1_struct[3], p1_struct[4]
                f_name1 = p1+'_'+pdb+'_'+chain+'+'+p1+'_'+pdb+'_'+chain
                f_name2 = p1+'_'+pdb+'_'+chain
            else:
                assert p1_struct[0] == 'AlphaFold'
                f_name1 = p1+'_AF+'+p1+'_AF'
                f_name2 = p1+'_AF'
            struct_file1 = './data/equidock_preds/'+f_name1+'.pdb'
            struct_file2 = './data/structs/'+f_name2+'.pdb'
            complex_file = './data/equidock_pred_complexes/'+f_name1+'_complex.pdb'
            uniprot_range, pdb_range = p1_struct[1], p1_struct[2]
            resmapping = dict(zip(unzip_res_range(pdb_range), unzip_res_range(uniprot_range)))
            to_complex(struct_file1, struct_file2, complex_file)
            ires = calc_ires(complex_file, 'A', 'B', p1, p2, resmapping, resmapping, len(fasta_dict[p1]), len(fasta_dict[p2]))

            with open('./data/equidock_ires/'+f_name1+'.pkl', 'wb') as outfile:
                pickle.dump(ires, outfile)

    else:
        p1_structs = structures[p1+'_'+p2][p1]
        p2_structs = structures[p1+'_'+p2][p2]

        if p1_structs[0][0] == 'PDB':
            assert p1_structs[0][4] != ' '
            pdb, chain = p1_structs[0][3], p1_structs[0][4]
            p1_file0 = p1+'_'+pdb+'_'+chain
        else:
            assert p1_structs[0][0] == 'AlphaFold'
            p1_file0 = p1+'_AF'

        uniprot_range, pdb_range = p1_structs[0][1], p1_structs[0][2]
        p1_file0_resmapping = dict(zip(unzip_res_range(pdb_range), unzip_res_range(uniprot_range)))

        if p2_structs[0][0] == 'PDB':
            assert p2_structs[0][4] != ' '
            pdb, chain = p2_structs[0][3], p2_structs[0][4]
            p2_file0 = p2+'_'+pdb+'_'+chain
        else:
            assert p2_structs[0][0] == 'AlphaFold'
            p2_file0 = p2+'_AF'

        uniprot_range, pdb_range = p2_structs[0][1], p2_structs[0][2]
        p2_file0_resmapping = dict(zip(unzip_res_range(pdb_range), unzip_res_range(uniprot_range)))

        for i, p1_struct in enumerate(p1_structs):
            if p1_struct[0] == 'PDB':
                assert p1_struct[4] != ' '
                pdb, chain = p1_struct[3], p1_struct[4]
                f_name1 = p1+'_'+pdb+'_'+chain+'+'+p2_file0
            else:
                assert p1_struct[0] == 'AlphaFold'
                f_name1 = p1+'_AF+'+p2_file0
            f_name2 = p2_file0

            struct_file1 = './data/equidock_preds/'+f_name1+'.pdb'
            struct_file2 = './data/structs/'+f_name2+'.pdb'
            complex_file = './data/equidock_pred_complexes/'+f_name1+'_complex.pdb'
            uniprot_range, pdb_range = p1_struct[1], p1_struct[2]
            resmapping = dict(zip(unzip_res_range(pdb_range), unzip_res_range(uniprot_range)))
            to_complex(struct_file1, struct_file2, complex_file)
            ires = calc_ires(complex_file, 'A', 'B', p1, p2, resmapping, p2_file0_resmapping, len(fasta_dict[p1]), len(fasta_dict[p2]))

            with open('./data/equidock_ires/'+f_name1+'.pkl', 'wb') as outfile:
                pickle.dump(ires, outfile)

        for i, p2_struct in enumerate(p2_structs[1:]):
            if p2_struct[0] == 'PDB':
                assert p2_struct[4] != ' '
                pdb, chain = p2_struct[3], p2_struct[4]
                f_name1 = p1_file0+'+'+p2+'_'+pdb+'_'+chain
                f_name2 = p2+'_'+pdb+'_'+chain
            else:
                assert p2_struct[0] == 'AlphaFold'
                f_name1 = p1_file0+'+'+p2+'_AF'
                f_name2 = p2+'_AF'

            struct_file1 = './data/equidock_preds/'+f_name1+'.pdb'
            struct_file2 = './data/structs/'+f_name2+'.pdb'
            complex_file = './data/equidock_pred_complexes/'+f_name1+'_complex.pdb'
            uniprot_range, pdb_range = p2_struct[1], p2_struct[2]
            resmapping = dict(zip(unzip_res_range(pdb_range), unzip_res_range(uniprot_range)))
            to_complex(struct_file1, struct_file2, complex_file)
            ires = calc_ires(complex_file, 'A', 'B', p1, p2, p1_file0_resmapping, resmapping, len(fasta_dict[p1]), len(fasta_dict[p2]))

            with open('./data/equidock_ires/'+f_name1+'.pkl', 'wb') as outfile:
                pickle.dump(ires, outfile)

print('done!')
