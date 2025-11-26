import os, sys, pickle
import numpy as np

def parse_dca(len1, len2, dca_file):
    coev_scores = np.empty((len1, len2))
    coev_scores[:] = np.nan
    
    with open(dca_file, 'r') as infile:
        lines = [line for line in infile if line[0] != '#']
        
    assert (len1+len2)*(len1+len2-1)/2 == len(lines)
    for line in lines:
        line_list = line.strip().split()
        assert int(line_list[0]) < int(line_list[1])
        id1, id2, score = int(line_list[0]), int(line_list[1]), float(line_list[2])
        if id1 <= len1 and id2 > len1:
            coev_scores[id1-1, id2-len1-1] = score
    assert not np.isnan(coev_scores).any()

    joined_dca = {}
    p1_max = np.nanmax(coev_scores, axis=1)
    p2_max = np.nanmax(coev_scores, axis=0)
    if p1 != p2:
        joined_dca['max'] = [p1_max, p2_max]
    else:
        p1_max = np.maximum(p1_max, p2_max)
        joined_dca['max'] = [p1_max, p1_max]

    p1_mean = np.nanmean(coev_scores, axis=1)
    p2_mean = np.nanmean(coev_scores, axis=0)
    if p1 != p2:
        joined_dca['mean'] = [p1_mean, p2_mean]
    else:
        p1_mean = np.maximum(p1_mean, p2_mean)
        joined_dca['mean'] = [p1_mean, p1_mean]

    p1_top10 = np.mean(np.sort(coev_scores, axis=1)[:,-10:], axis=1)
    p2_top10 = np.mean(np.sort(coev_scores, axis=0)[-10:,:], axis=0)
    if p1 != p2:
        joined_dca['top10'] = [p1_top10, p2_top10]
    else:
        p1_top10 = np.maximum(p1_top10, p2_top10)
        joined_dca['top10'] = [p1_top10, p1_top10]
    return joined_dca


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

interaction_len = {}
for interaction in interactions:
    p1, p2 = interaction
    assert p1 <= p2
    interaction_len[interaction] = len(fasta_dict[p1])+len(fasta_dict[p2])

interaction_len = sorted(interaction_len.items(), key=lambda x: x[1])
interactions = [ele[0] for ele in interaction_len]

for interaction in interactions:
    p1, p2 = interaction
    plmdca_file = './data/joined_raw_plmdca/PLMDCA_apc_fn_scores_'+p1+'_'+p2+'.txt'
    if os.path.exists(plmdca_file):
        joined_plmdca = parse_dca(len(fasta_dict[p1]), len(fasta_dict[p2]), plmdca_file)
    else:
        joined_plmdca = {}
        joined_plmdca['max'] = [[np.nan for i in range(len(fasta_dict[p1]))], [np.nan for i in range(len(fasta_dict[p2]))]]
        joined_plmdca['mean'] = [[np.nan for i in range(len(fasta_dict[p1]))], [np.nan for i in range(len(fasta_dict[p2]))]]
        joined_plmdca['top10'] = [[np.nan for i in range(len(fasta_dict[p1]))], [np.nan for i in range(len(fasta_dict[p2]))]]
        
    with open('./data/joined_plmdca/'+p1+'_'+p2+'.pkl', 'wb') as outfile:
        pickle.dump(joined_plmdca, outfile)
            
    mfdca_file = './data/joined_raw_mfdca/MFDCA_apc_fn_scores_'+p1+'_'+p2+'.txt'
    if os.path.exists(mfdca_file):
        joined_mfdca = parse_dca(len(fasta_dict[p1]), len(fasta_dict[p2]), mfdca_file)
    else:
        joined_mfdca = {}
        joined_mfdca['max'] = [[np.nan for i in range(len(fasta_dict[p1]))], [np.nan for i in range(len(fasta_dict[p2]))]]
        joined_mfdca['mean'] = [[np.nan for i in range(len(fasta_dict[p1]))], [np.nan for i in range(len(fasta_dict[p2]))]]
        joined_mfdca['top10'] = [[np.nan for i in range(len(fasta_dict[p1]))], [np.nan for i in range(len(fasta_dict[p2]))]]
        
    with open('./data/joined_mfdca/'+p1+'_'+p2+'.pkl', 'wb') as outfile:
        pickle.dump(joined_mfdca, outfile)
        
print('done!')
