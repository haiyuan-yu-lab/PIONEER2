import os, pickle
import numpy as np

def parse_dca(prot_len, dca_file):
    coev_scores = np.empty((prot_len, prot_len))
    coev_scores[:] = np.nan
    
    with open(dca_file, 'r') as infile:
        lines = [line for line in infile if line[0] != '#']
        
    assert (prot_len)*(prot_len-1)/2 == len(lines)
    for line in lines:
        line_list = line.strip().split()
        assert int(line_list[0]) < int(line_list[1])
        id1, id2, score = int(line_list[0]), int(line_list[1]), float(line_list[2])
        coev_scores[id1-1, id2-1] = score
        coev_scores[id2-1, id1-1] = score
    for i in range(prot_len):
        for j in range(prot_len):
            if i != j:
                assert not np.isnan(coev_scores[i, j])
            else:
                assert np.isnan(coev_scores[i, j])

    dca_node = {}
    dca_node['max'] = np.nanmax(coev_scores, axis=1)
    dca_node['mean'] = np.nanmean(coev_scores, axis=1)
    dca_node['top10'] = np.mean(np.sort(coev_scores, axis=1)[:,-11:-1], axis=1)
    return dca_node, coev_scores


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
    plmdca_file = './data/single_raw_plmdca/PLMDCA_apc_fn_scores_'+identifier+'.txt'
    if os.path.exists(plmdca_file):
        plmdca_node, plmdca_scores = parse_dca(len(fasta_dict[identifier]), plmdca_file)
    else:
        plmdca_node = {}
        plmdca_node['max'] = [np.nan for i in range(len(fasta_dict[identifier]))]
        plmdca_node['mean'] = [np.nan for i in range(len(fasta_dict[identifier]))]
        plmdca_node['top10'] = [np.nan for i in range(len(fasta_dict[identifier]))]
        
        plmdca_scores = np.empty((len(fasta_dict[identifier]), len(fasta_dict[identifier])))
        plmdca_scores[:] = np.nan
        
    with open('./data/single_plmdca/'+identifier+'.pkl', 'wb') as outfile:
        pickle.dump([plmdca_node, plmdca_scores], outfile)
            
    mfdca_file = './data/single_raw_mfdca/MFDCA_apc_fn_scores_'+identifier+'.txt'
    if os.path.exists(mfdca_file):
        mfdca_node, mfdca_scores = parse_dca(len(fasta_dict[identifier]), mfdca_file)
    else:
        mfdca_node = {}
        mfdca_node['max'] = [np.nan for i in range(len(fasta_dict[identifier]))]
        mfdca_node['mean'] = [np.nan for i in range(len(fasta_dict[identifier]))]
        mfdca_node['top10'] = [np.nan for i in range(len(fasta_dict[identifier]))]
        
        mfdca_scores = np.empty((len(fasta_dict[identifier]), len(fasta_dict[identifier])))
        mfdca_scores[:] = np.nan
        
    with open('./data/single_mfdca/'+identifier+'.pkl', 'wb') as outfile:
        pickle.dump([mfdca_node, mfdca_scores], outfile)
        
print('done!')
