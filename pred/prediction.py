import os, sys, pickle, argparse, json, copy
import numpy as np
import pandas as pd
import torch, dgl
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from data_generator import load_dataset
from network import Network
from metrics import *

with open('config.json') as f:
    config = json.load(f)

config['device'] = torch.device('cuda:0')
net_params = config['net_params']
net_params['init_lr'] = 0.0001
net_params['hidden_dim'] = 128
net_params['clip'] = True

interactions_to_pred = []
with open(sys.argv[1], 'r') as infile:
    for line in infile:
        line_list = line.strip().split()
        interactions_to_pred.append((line_list[0], line_list[1]))
result_folder = sys.argv[2]

graph_feature_dir = '../data/struct_graphs_input'
gmm_feature_dir = '../data/struct_graphs_gmms_input'
        
fasta_dict = {}
with open('../data/all_preppi_uniprot_seqs.txt', 'r') as infile:
    for line in infile:
        line_list = line.strip().split()
        fasta_dict[line_list[0]] = line_list[1]

with open('../data/structs_to_use.json', 'r') as infile:
    structures = json.load(infile)

all_interactions = []
for interaction in sorted(list(structures.keys())):
    p1, p2 = interaction.split('_')
    all_interactions.append((p1, p2))
print(len(all_interactions))

with open('./mean_values.pkl', 'rb') as infile:
    mean_values = pickle.load(infile)
    
with open('./min_values.pkl', 'rb') as infile:
    min_values = pickle.load(infile)
    
with open('./max_values.pkl', 'rb') as infile:
    max_values = pickle.load(infile)

net_params['in_dim_node'] = len(mean_values[0])+net_params['in_dim_gmm']*2
net_params['in_dim_edge'] = len(mean_values[1])
net_params['batch_size'] = 1

model = Network(config)
model = model.to(config['device'])
model.load_state_dict(torch.load('model.dat', map_location=config['device']))
model.eval()

for interaction in interactions_to_pred:
    struct_sources = {}
    struct_sources[interaction[0]] = set()
    struct_sources[interaction[1]] = set()
    for e in structures[interaction[0]+'_'+interaction[1]][interaction[0]]:
        struct_sources[interaction[0]].add(e[0])
    for e in structures[interaction[0]+'_'+interaction[1]][interaction[1]]:
        struct_sources[interaction[1]].add(e[0])
        
    pred_dataset = load_dataset([interaction], config['net_params']['batch_size'], 
                                graph_feature_dir, gmm_feature_dir, net_params, mean_values, 
                                min_values, max_values, net_params['frame_aa_Kmax'], net_params['frame_atom_Kmax'], 
                                shuffle=False)
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(pred_dataset):
            target_batch_graphs = []
            target_batch_aa_gmms = []
            target_batch_atom_gmms = []
            target_batch_atom_nums = []
            batch_seq_ids = []
            batch_lens = []
            partner_batch_graphs = []
            partner_batch_aa_gmms = []
            partner_batch_atom_gmms = []
            partner_batch_atom_nums = []
            batch_nums = []
            batch_interactions = []
            for ele in batch_data:
                target_batch_graphs.append(ele[0])
                batch_seq_ids.append(ele[1])
                batch_lens.append(ele[2])
                partner_batch_graphs.append(ele[3])
                batch_nums.append(ele[4])
                target_batch_aa_gmms.append(ele[5])
                target_batch_atom_gmms.append(ele[6])
                target_batch_atom_nums.append(ele[7])
                partner_batch_aa_gmms.append(ele[8])
                partner_batch_atom_gmms.append(ele[9])
                partner_batch_atom_nums.append(ele[10])
                batch_interactions.append(ele[11])

            target_batch_graphs = dgl.batch(target_batch_graphs).to(config['device'])
            target_batch_aa_gmms = torch.cat(target_batch_aa_gmms, 0).to(config['device'])
            target_batch_atom_gmms = torch.cat(target_batch_atom_gmms, 0).to(config['device'])
            partner_batch_graphs = dgl.batch(partner_batch_graphs).to(config['device'])
            partner_batch_aa_gmms = torch.cat(partner_batch_aa_gmms, 0).to(config['device'])
            partner_batch_atom_gmms = torch.cat(partner_batch_atom_gmms, 0).to(config['device'])

            if config['net_params']['lap_pos_enc']:
                target_batch_lap_pos_enc = target_batch_graphs.ndata['lap_pos_enc']
                partner_batch_lap_pos_enc = partner_batch_graphs.ndata['lap_pos_enc']
            else:
                target_batch_lap_pos_enc = None
                partner_batch_lap_pos_enc = None

            if config['net_params']['wl_pos_enc']:
                target_batch_wl_pos_enc = target_batch_graphs.ndata['wl_pos_enc']
                partner_batch_wl_pos_enc = partner_batch_graphs.ndata['wl_pos_enc']
            else:
                target_batch_wl_pos_enc = None
                partner_batch_wl_pos_enc = None

            batch_scores = model.forward(target_batch_graphs, target_batch_aa_gmms, target_batch_atom_gmms, copy.deepcopy(target_batch_atom_nums), target_batch_lap_pos_enc, target_batch_wl_pos_enc, partner_batch_graphs, partner_batch_aa_gmms, partner_batch_atom_gmms, copy.deepcopy(partner_batch_atom_nums), partner_batch_lap_pos_enc, partner_batch_wl_pos_enc, batch_nums)
            shapes = batch_scores.size()
            batch_scores = batch_scores.view(shapes[0]*shapes[1])

            assert batch_scores.size()[0] == np.sum([ele[-1][1]+1 for ele in batch_nums])
            start, end = 0, batch_nums[0][-1][1]
            batch_scores_ = [batch_scores[start:end+1]]
            assert len(batch_nums) == 1
            for i in range(1, len(batch_nums)):
                start = end+1
                end += batch_nums[i][-1][1]+1
                batch_scores_.append(batch_scores[start:end+1])
                
            assert len(batch_seq_ids) == 1
            assert len(batch_scores_) == 1
            for i in range(len(batch_seq_ids)):
                batch_seq_ids_ = batch_seq_ids[i]
                assert len(batch_seq_ids_) == batch_nums[i][-1][1]+1
                batch_seq_id_indexes = [j for j in range(len(batch_seq_ids_)) if batch_seq_ids_[j] not in batch_seq_ids_[:j]]
                batch_scores_[i] = batch_scores_[i][batch_seq_id_indexes]
                batch_seq_ids[i] = [batch_seq_ids_[j] for j in batch_seq_id_indexes]

            if config['device'].type == 'cuda':
                for i in range(len(batch_scores_)):
                    batch_scores_[i] = batch_scores_[i].detach().cpu().numpy()
            else:
                for i in range(len(batch_scores_)):
                    batch_scores_[i] = batch_scores_[i].detach().numpy()
                    
            assert len(batch_interactions) == 1
            for i in range(len(batch_seq_ids)):
                batch_seq_ids_ = copy.deepcopy(batch_seq_ids[i])
                batch_seq_ids_ = sorted(batch_seq_ids_)
                interaction_name = batch_interactions[i]
                
                tmp = {}
                tmp['interaction'] = []
                tmp['index'] = []
                tmp['prob'] = []
                for j in range(len(batch_seq_ids_)):
                    tmp['interaction'].append(interaction_name)
                    tmp['index'].append(batch_seq_ids_[j])
                    tmp['prob'].append(batch_scores_[i][j])
                    
                if len(tmp['index']) != len(fasta_dict[interaction_name.split('_')[0]]):
                    assert 'AlphaFold' not in struct_sources[interaction_name.split('_')[0]]
                    
                df = pd.DataFrame(tmp)
                with open(os.path.join(result_folder, interaction_name+'.pkl'), 'wb') as outfile:
                    pickle.dump(df, outfile)
                    
print('done!')
