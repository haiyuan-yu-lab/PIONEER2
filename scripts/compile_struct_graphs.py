import os, sys, json, pickle, itertools, copy, torch, dgl
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from utils import unzip_res_range

def normalize(features):
    return (features - np.nanmean(features)) / np.nanstd(features)

def get_ss(index, pdb_index_file1, pdb_index_file2):
    if index == 0:
        pdb_index_file = pdb_index_file1
    else:
        if index != 1:
            sys.exit('Invalid index: ', index)
        pdb_index_file = pdb_index_file2
        
    ss = np.load('./data/ss/'+pdb_index_file+'.npy')
    return ss

def get_rd(index, pdb_index_file1, pdb_index_file2):
    if index == 0:
        pdb_index_file = pdb_index_file1
    else:
        if index != 1:
            sys.exit('Invalid index: ', index)
        pdb_index_file = pdb_index_file2
        
    rd = np.load('./data/rd/'+pdb_index_file+'.npy')
    return rd

def get_sasa(index, pdb_index_file1, pdb_index_file2):
    if index == 0:
        pdb_index_file = pdb_index_file1
    else:
        if index != 1:
            sys.exit('Invalid index: ', index)
        pdb_index_file = pdb_index_file2
        
    sasa = np.load('./data/sasa/'+pdb_index_file+'.npy')
    return sasa

def get_dock(index, pdb_file_index1, pdb_file_index2):
    with open('./data/equidock_ires/'+pdb_file_index1+'+'+pdb_file_index2+'.pkl', 'rb') as infile:
        dock = pickle.load(infile)
    return dock[index]

def get_preppi(p1, p2, index, preppi_info):
    assert p1 <= p2
    if index == 0:
        if (p1, p2) in preppi_info:
            ids = preppi_info[(p1, p2)][0][0]
            feat = preppi_info[(p1, p2)][1][0]
        else:
            ids = None
            feat = None
    else:
        if index != 1:
            sys.exit('Invalid index: ', index)
            
        if (p1, p2) in preppi_info:
            ids = preppi_info[(p1, p2)][0][1]
            feat = preppi_info[(p1, p2)][1][1]
        else:
            ids = None
            feat = None
    return ids, feat

def compile_node_feature(p1, p2, index, pdb_index_file1, pdb_index_file2, preppi_info):
    if index == 0:
        p = p1
    else:
        if index != 1:
            sys.exit('Invalid index to indicate p1 or p2:', index)
        p = p2
        
    with open('./data/expasy/'+p+'.pkl', 'rb') as infile:
        expasy = pickle.load(infile)
    node_feature = [expasy['ACCE']]
    node_feature.append(expasy['AREA'])
    node_feature.append(expasy['BULK'])
    node_feature.append(expasy['COMP'])
    node_feature.append(expasy['HPHO'])
    node_feature.append(expasy['POLA'])
    node_feature.append(expasy['TRAN'])

    js = np.load('./data/js_uniref90/'+p+'.npy')
    node_feature.append(js)
        
    with open('./data/joined_plmdca/'+p1+'_'+p2+'.pkl', 'rb') as infile:
        joined_plmdca = pickle.load(infile)
    node_feature.append(joined_plmdca['max'][index])
    node_feature.append(joined_plmdca['mean'][index])
    node_feature.append(joined_plmdca['top10'][index])
    
    with open('./data/joined_mfdca/'+p1+'_'+p2+'.pkl', 'rb') as infile:
        joined_mfdca = pickle.load(infile)
    node_feature.append(joined_mfdca['max'][index])
    node_feature.append(joined_mfdca['mean'][index])
    node_feature.append(joined_mfdca['top10'][index])
    
    with open('./data/pssm_uniref90/'+p+'_ori.pkl', 'rb') as infile:
        pssm = pickle.load(infile).to_numpy()
    node_feature = np.concatenate((np.array(node_feature).T, pssm), axis=1)
    
    ss = get_ss(index, pdb_index_file1, pdb_index_file2)
    node_feature = np.concatenate((node_feature, ss), axis=1)
    
    rd = get_rd(index, pdb_index_file1, pdb_index_file2)
    node_feature = np.column_stack((node_feature, rd))
    
    sasa = get_sasa(index, pdb_index_file1, pdb_index_file2)
    node_feature = np.column_stack((node_feature, sasa))
    
    dock = get_dock(index, pdb_index_file1, pdb_index_file2)
    node_feature = np.column_stack((node_feature, dock))
    
    ids, feat = get_preppi(p1, p2, index, preppi_info)
    preppi_feat = np.array([ [0, 1, 1, 0, 0, 0, 0, 0] for i in range(node_feature.shape[0]) ])
    if ids:
        preppi_feat[ids] = feat
    node_feature = np.concatenate((node_feature, preppi_feat), axis=1)
    return node_feature

def pdb_txt2array(pdb_file):
    with open(pdb_file, 'r') as infile:
        lines = infile.readlines()
        
    pdb_info = []
    for line in lines:
        line = line.strip()
        line_list = [line[0:5], line[6:11], line[12:16], line[16], line[17:20], line[21], line[22:27], line[30:38], line[38:46], line[46:54]]
        pdb_info.append([i.strip() for i in line_list])
    return np.array(pdb_info, dtype='str')

def get_residue_info(pdb_array):
    atom_res_array = pdb_array[:,6]
    boundary_list = []
    start_pointer = 0
    curr_pointer = 0
    curr_atom = atom_res_array[0]
    
    while(curr_pointer < atom_res_array.shape[0] - 1):
        curr_pointer += 1
        if atom_res_array[curr_pointer] != curr_atom:
            boundary_list.append([start_pointer, curr_pointer - 1])
            start_pointer = curr_pointer
            curr_atom = atom_res_array[curr_pointer]
    boundary_list.append([start_pointer, atom_res_array.shape[0] - 1])
    return np.array(boundary_list)

def get_distance_matrix(pdb_array, residue_index, distance_type):
    if distance_type == 'c_alpha':
        coord_array = np.empty((residue_index.shape[0], 3))
        for i in range(residue_index.shape[0]):
            res_start, res_end = residue_index[i]
            
            flag = False
            res_array = pdb_array[res_start:res_end+1]
            for j in range(res_array.shape[0]):
                if res_array[j][2] == 'CA':
                    coord_array[i] = res_array[:,7:10][j].astype(np.float64)
                    flag = True
                    break
            if not flag:
                coord_i = pdb_array[:,7:10][res_start:res_end+1].astype(np.float64)
                coord_array[i] = np.mean(coord_i, axis=0)
        residue_dm = squareform(pdist(coord_array))
        
    elif distance_type == 'centroid':
        coord_array = np.empty((residue_index.shape[0], 3))
        for i in range(residue_index.shape[0]):
            res_start, res_end = residue_index[i]
            coord_i = pdb_array[:,7:10][res_start:res_end+1].astype(np.float64)
            coord_array[i] = np.mean(coord_i, axis=0)
        residue_dm = squareform(pdist(coord_array))
        
    elif distance_type == 'atoms_average':
        full_atom_dist = squareform(pdist(pdb_array[:,7:10].astype(float)))
        residue_dm = np.zeros((residue_index.shape[0], residue_index.shape[0]))
        for i, j in itertools.combinations(range(residue_index.shape[0]), 2):
            index_i = residue_index[i]
            index_j = residue_index[j]
            distance_ij = np.mean(full_atom_dist[index_i[0]:index_i[1]+1,index_j[0]:index_j[1]+1])
            residue_dm[i][j] = distance_ij
            residue_dm[j][i] = distance_ij
            
    else:
        raise ValueError('Invalid distance type: %s' % distance_type)
    return residue_dm

def get_neighbor_index(residue_dm, num_neighbors):
    return residue_dm.argsort()[:, 1:num_neighbors+1]

def get_normal(acid_plane):
    cp = np.cross(acid_plane[2] - acid_plane[1], acid_plane[0] - acid_plane[1])
    if np.all(cp == 0):
        return np.array([np.nan] * 3)
    normal = cp/np.linalg.norm(cp,2)
    return normal

def fill_nan_mean(array, axis=0):
    if axis not in [0, 1]:
        raise ValueError('Invalid axis: %s' % axis)
    mean_array = np.nanmean(array, axis=axis)
    inds = np.where(np.isnan(array))
    array[inds] = np.take(mean_array, inds[1-axis])
    if np.any(np.isnan(array)):
        full_array_mean = np.nanmean(array)
        inds = np.unique(np.where(np.isnan(array))[1-axis])
        if axis == 0:
            array[:,inds] = full_array_mean
        else:
            array[inds] = full_array_mean
    return array

def get_neighbor_angle(pdb_array, residue_index, neighbor_index):
    normal_vector_array = np.empty((neighbor_index.shape[0], 3), dtype=np.float64)
    for i, (res_start, res_end) in enumerate(residue_index):
        res_info = pdb_array[res_start:res_end+1]
        res_acid_plane_index = np.where(np.logical_and(np.isin(res_info[:,2], ['CA', 'C', 'O']), np.isin(res_info[:,3], ['', 'A'])))
        res_acid_plane = res_info[res_acid_plane_index][:,7:10].astype(np.float64)
        if res_acid_plane.shape[0] != 3:
            normal_vector_array[i] = np.array([np.nan] * 3)
            continue
        normal_vector = get_normal(res_acid_plane)
        if np.all(np.isnan(normal_vector)):
            normal_vector_array[i] = np.array([np.nan] * 3)
        else:
            normal_vector_array[i] = normal_vector
            
    pairwise_normal_dot = normal_vector_array.dot(normal_vector_array.T)
    pairwise_normal_dot[pairwise_normal_dot > 1] = 1
    pairwise_normal_dot[pairwise_normal_dot < -1] = -1
    
    pairwise_angle = np.arccos(pairwise_normal_dot) / np.pi
    
    angle_matrix = np.empty_like(neighbor_index, dtype=np.float64)
    for i, index in enumerate(neighbor_index):
        angle_matrix[i] = pairwise_angle[i, index]
    
    angle_matrix = fill_nan_mean(angle_matrix, axis=0)
    return angle_matrix

def get_edge_data(residue_dm, neighbor_index, neighbor_angle):
    edge_matrix = np.zeros((neighbor_index.shape[0], neighbor_index.shape[1], 2))
    for i, dist in enumerate(residue_dm):
        edge_matrix[i][:,0] = dist[neighbor_index[i]]
        edge_matrix[i][:,1] = neighbor_angle[i]
    return edge_matrix

def compile_edge_feature(pdb_array, p, seq_id, num_neighbors=16, distance_type='c_alpha'):
    residue_index = get_residue_info(pdb_array)
    residue_dm = get_distance_matrix(pdb_array, residue_index, distance_type)
    neighbor_index = get_neighbor_index(residue_dm, num_neighbors)
    neighbor_angle = get_neighbor_angle(pdb_array, residue_index, neighbor_index)
    edge_data = get_edge_data(residue_dm, neighbor_index, neighbor_angle)
    
    source = np.reshape(neighbor_index, (-1, 1)).squeeze(axis=1)
    target = np.repeat(np.arange(residue_index.shape[0]), neighbor_index.shape[1])
    edge_index = [source, target]
    edge_feature = np.reshape(edge_data, (residue_index.shape[0] * neighbor_index.shape[1], -1))
    
    dca_scores = []
    with open('./data/single_plmdca/'+p+'.pkl', 'rb') as infile:
        plmdca = pickle.load(infile)[1]
    dca_scores.append(plmdca)
    with open('./data/single_mfdca/'+p+'.pkl', 'rb') as infile:
        mfdca = pickle.load(infile)[1]
    dca_scores.append(mfdca)
    
    dca_feature = []
    for i in range(source.shape[0]):
        tmp = []
        for m in dca_scores:
            tmp.append(m[seq_id[source[i]], seq_id[target[i]]])
        dca_feature.append(tmp)
        
    edge_feature = np.concatenate((edge_feature, np.array(dca_feature)), axis=1)
    return edge_index, edge_feature

def compile_graphs(p1, p2, p_index, structs1, structs2, len_p1, len_p2, struct_dir, gmm_dir, preppi_info):
    if p1 == p2:
        target_batch_graphs, target_batch_aa_gmms, target_batch_atom_gmms, target_batch_atom_nums, batch_seq_ids, partner_batch_graphs, partner_batch_aa_gmms, partner_batch_atom_gmms, partner_batch_atom_nums, batch_nums = [], [], [], [], [], [], [], [], [], []
        for i, struct in enumerate(structs1):
            if struct[0] == 'PDB':
                assert struct[4] != ' '
                pdb_file = os.path.join(struct_dir, p1+'_'+struct[3]+'_'+struct[4]+'.pdb')
            else:
                assert struct[0] == 'AlphaFold'
                pdb_file = os.path.join(struct_dir, p1+'_AF.pdb')
            
            pdb_array = pdb_txt2array(pdb_file)
            pdb_array = pdb_array[(np.where((pdb_array[:,3] == '') | (pdb_array[:,3] == 'A')))]
            res_id = np.sort(np.unique(pdb_array[:, 6]))
            
            uniprot_range, pdb_range = struct[1], struct[2]
            mapping = dict(zip(unzip_res_range(pdb_range), unzip_res_range(uniprot_range)))
            seq_id = sorted([int(mapping[i])-1 for i in res_id])
            
            node_feature = compile_node_feature(p1, p2, 0, os.path.basename(pdb_file)[:-4], os.path.basename(pdb_file)[:-4], preppi_info)
            
            edge_index, edge_feature = compile_edge_feature(pdb_array, p1, seq_id, 16)
            
            graph = dgl.graph((torch.tensor(edge_index[0]), torch.tensor(edge_index[1])))
            graph.ndata['feat'] = torch.from_numpy(node_feature[seq_id, :])
            graph.edata['feat'] = torch.from_numpy(edge_feature)
            
            with open(os.path.join(gmm_dir, os.path.basename(pdb_file)[:-4]+'.pkl'), 'rb') as infile:
                gmm = pickle.load(infile)
            assert len(gmm['atom_nums']) == len(res_id)
            
            target_batch_graphs.append(graph)
            target_batch_aa_gmms.append(torch.tensor(gmm['gmm_activities'][0]))
            target_batch_atom_gmms.append(torch.tensor(gmm['gmm_activities'][1]))
            target_batch_atom_nums.append(gmm['atom_nums'])
            batch_seq_ids += seq_id
            partner_batch_graphs.append(copy.deepcopy(graph))
            partner_batch_aa_gmms.append(torch.tensor(copy.deepcopy(gmm['gmm_activities'][0])))
            partner_batch_atom_gmms.append(torch.tensor(copy.deepcopy(gmm['gmm_activities'][1])))
            partner_batch_atom_nums.append(copy.deepcopy(gmm['atom_nums']))
            
            if not batch_nums:
                batch_nums.append([0, len(seq_id)-1, 0, len(seq_id)-1])
            else:
                batch_nums.append([batch_nums[-1][1]+1, batch_nums[-1][1]+len(seq_id), batch_nums[-1][1]+1, batch_nums[-1][1]+len(seq_id)])

        target_batch_graphs = dgl.batch(target_batch_graphs)
        partner_batch_graphs = dgl.batch(partner_batch_graphs)
        return [target_batch_graphs, batch_seq_ids, len_p1, partner_batch_graphs, batch_nums], [target_batch_aa_gmms, target_batch_atom_gmms, target_batch_atom_nums, partner_batch_aa_gmms, partner_batch_atom_gmms, partner_batch_atom_nums]
            
    else:
        target_batch_graphs, target_batch_aa_gmms, target_batch_atom_gmms, target_batch_atom_nums, batch_seq_ids, partner_batch_graphs, partner_batch_aa_gmms, partner_batch_atom_gmms, partner_batch_atom_nums, batch_nums = [], [], [], [], [], [], [], [], [], []
        if p_index == 0:
            target_structs = structs1
            partner_struct = structs2[0]
            target_p = p1
            partner_p = p2
            if partner_struct[0] == 'PDB':
                assert partner_struct[4] != ' '
                partner_pdb_file = os.path.join(struct_dir, p2+'_'+partner_struct[3]+'_'+partner_struct[4]+'.pdb')
            else:
                assert partner_struct[0] == 'AlphaFold'
                partner_pdb_file = os.path.join(struct_dir, p2+'_AF.pdb')
            len_p = len_p1
        else:
            if p_index != 1:
                sys.exit('Invalid p_index to compile graphs:', p_index)
            target_structs = structs2
            partner_struct = structs1[0]
            target_p = p2
            partner_p = p1
            if partner_struct[0] == 'PDB':
                assert partner_struct[4] != ' '
                partner_pdb_file = os.path.join(struct_dir, p1+'_'+partner_struct[3]+'_'+partner_struct[4]+'.pdb')
            else:
                assert partner_struct[0] == 'AlphaFold'
                partner_pdb_file = os.path.join(struct_dir, p1+'_AF.pdb')
            len_p = len_p2
            
        partner_pdb_array = pdb_txt2array(partner_pdb_file)
        partner_pdb_array = partner_pdb_array[(np.where((partner_pdb_array[:,3] == '') | (partner_pdb_array[:,3] == 'A')))]
        partner_res_id = np.sort(np.unique(partner_pdb_array[:, 6]))

        partner_uniprot_range, partner_pdb_range = partner_struct[1], partner_struct[2]
        partner_mapping = dict(zip(unzip_res_range(partner_pdb_range), unzip_res_range(partner_uniprot_range)))
        partner_seq_id = sorted([int(partner_mapping[i])-1 for i in partner_res_id])

        partner_edge_index, partner_edge_feature = compile_edge_feature(partner_pdb_array, partner_p, partner_seq_id, 16)

        partner_graph = dgl.graph((torch.tensor(partner_edge_index[0]), torch.tensor(partner_edge_index[1])))
        partner_graph.edata['feat'] = torch.from_numpy(partner_edge_feature)
        
        with open(os.path.join(gmm_dir, os.path.basename(partner_pdb_file)[:-4]+'.pkl'), 'rb') as infile:
            partner_gmm = pickle.load(infile)
        assert len(partner_gmm['atom_nums']) == len(partner_res_id)
        
        for i, target_struct in enumerate(target_structs):
            if target_struct[0] == 'PDB':
                assert target_struct[4] != ' '
                if p_index == 0:
                    target_pdb_file = os.path.join(struct_dir, p1+'_'+target_struct[3]+'_'+target_struct[4]+'.pdb')
                else:
                    target_pdb_file = os.path.join(struct_dir, p2+'_'+target_struct[3]+'_'+target_struct[4]+'.pdb')
            else:
                assert target_struct[0] == 'AlphaFold'
                if p_index == 0:
                    target_pdb_file = os.path.join(struct_dir, p1+'_AF.pdb')
                else:
                    target_pdb_file = os.path.join(struct_dir, p2+'_AF.pdb')
                    
            target_pdb_array = pdb_txt2array(target_pdb_file)
            target_pdb_array = target_pdb_array[(np.where((target_pdb_array[:,3] == '') | (target_pdb_array[:,3] == 'A')))]
            target_res_id = np.sort(np.unique(target_pdb_array[:, 6]))
            
            target_uniprot_range, target_pdb_range = target_struct[1], target_struct[2]
            target_mapping = dict(zip(unzip_res_range(target_pdb_range), unzip_res_range(target_uniprot_range)))
            target_seq_id = sorted([int(target_mapping[i])-1 for i in target_res_id])
            
            if p_index == 0:
                target_node_feature = compile_node_feature(p1, p2, p_index, os.path.basename(target_pdb_file)[:-4], os.path.basename(partner_pdb_file)[:-4], preppi_info)
                
                partner_node_feature = compile_node_feature(p1, p2, 1, os.path.basename(target_pdb_file)[:-4], os.path.basename(partner_pdb_file)[:-4], preppi_info)
            else:
                target_node_feature = compile_node_feature(p1, p2, p_index, os.path.basename(partner_pdb_file)[:-4], os.path.basename(target_pdb_file)[:-4], preppi_info)
                
                partner_node_feature = compile_node_feature(p1, p2, 0, os.path.basename(partner_pdb_file)[:-4], os.path.basename(target_pdb_file)[:-4], preppi_info)
            
            target_edge_index, target_edge_feature = compile_edge_feature(target_pdb_array, target_p, target_seq_id, 16)

            target_graph = dgl.graph((torch.tensor(target_edge_index[0]), torch.tensor(target_edge_index[1])))
            target_graph.ndata['feat'] = torch.from_numpy(target_node_feature[target_seq_id, :])
            target_graph.edata['feat'] = torch.from_numpy(target_edge_feature)
            
            partner_graph.ndata['feat'] = torch.from_numpy(partner_node_feature[partner_seq_id, :])
            
            with open(os.path.join(gmm_dir, os.path.basename(target_pdb_file)[:-4]+'.pkl'), 'rb') as infile:
                target_gmm = pickle.load(infile)
            assert len(target_gmm['atom_nums']) == len(target_res_id)
            
            target_batch_graphs.append(target_graph)
            target_batch_aa_gmms.append(torch.tensor(target_gmm['gmm_activities'][0]))
            target_batch_atom_gmms.append(torch.tensor(target_gmm['gmm_activities'][1]))
            target_batch_atom_nums.append(target_gmm['atom_nums'])
            batch_seq_ids += target_seq_id
            partner_batch_graphs.append(copy.deepcopy(partner_graph))
            partner_batch_aa_gmms.append(torch.tensor(copy.deepcopy(partner_gmm['gmm_activities'][0])))
            partner_batch_atom_gmms.append(torch.tensor(copy.deepcopy(partner_gmm['gmm_activities'][1])))
            partner_batch_atom_nums.append(copy.deepcopy(partner_gmm['atom_nums']))
            
            if not batch_nums:
                batch_nums.append([0, len(target_seq_id)-1, 0, len(partner_seq_id)-1])
            else:
                batch_nums.append([batch_nums[-1][1]+1, batch_nums[-1][1]+len(target_seq_id), batch_nums[-1][3]+1, batch_nums[-1][3]+len(partner_seq_id)])
                
        target_batch_graphs = dgl.batch(target_batch_graphs)
        partner_batch_graphs = dgl.batch(partner_batch_graphs)
        return [target_batch_graphs, batch_seq_ids, len_p, partner_batch_graphs, batch_nums], [target_batch_aa_gmms, target_batch_atom_gmms, target_batch_atom_nums, partner_batch_aa_gmms, partner_batch_atom_gmms, partner_batch_atom_nums]


fasta_dict = {}
with open('./data/all_preppi_uniprot_seqs.txt', 'r') as infile:
    for line in infile:
        line_list = line.strip().split()
        fasta_dict[line_list[0]] = line_list[1]

with open('./data/structs_to_use.json', 'r') as infile:
    structures = json.load(infile)

all_interactions = sorted(list(structures.keys()))

preppi_info = {}
preppi_1 = pd.read_csv('./data/PrePPI_in_HINT_all.csv', sep=',')
preppi_1.rename(columns={'V7': 'LR', 'V9':'ifcs'}, inplace=True)
for index, row in preppi_1.iterrows():
    p1, p2 = row['V3'].split('_')
    assert p1 <= p2
    interfacial_contacts = [set(), set()]

    if isinstance(row['ifcs'], str):
        for e in row['ifcs'].split():
            e_ = e.split('-')
            if int(e_[0]) <= len(fasta_dict[p1]):
                interfacial_contacts[0].add(int(e_[0])-1)
            if int(e_[1]) <= len(fasta_dict[p2]):
                interfacial_contacts[1].add(int(e_[1])-1)
    interfacial_contacts[0] = sorted(list(interfacial_contacts[0]))
    interfacial_contacts[1] = sorted(list(interfacial_contacts[1]))

    if interfacial_contacts[0]:
        assert interfacial_contacts[0][-1] <= len(fasta_dict[p1])
    if interfacial_contacts[1]:
        assert interfacial_contacts[1][-1] <= len(fasta_dict[p2])

    assert not np.isnan(row['LR']) and isinstance(row['LR'], float)

    preppi_info[(p1, p2)] = [interfacial_contacts, [[row['LR']], [row['LR']]]]

preppi_2 = pd.read_csv('./data/hint_features.csv', sep=',')
for index, row in preppi_2.iterrows():
    p1, p2 = row['ppi'].split('_')
    assert p1 <= p2

    assert not np.isnan(row['psd1']) and isinstance(row['psd1'], float)
    assert not np.isnan(row['psd2']) and isinstance(row['psd2'], float)
    assert not np.isnan(row['no_residues']) and isinstance(row['no_residues'], int)
    assert not np.isnan(row['siz']) and isinstance(row['siz'], int)
    assert not np.isnan(row['cov']) and isinstance(row['cov'], float)
    assert not np.isnan(row['ol']) and isinstance(row['ol'], int)
    assert not np.isnan(row['os']) and isinstance(row['os'], int)

    e = preppi_info[(p1, p2)][0]
    e1 = preppi_info[(p1, p2)][1][0]+[row['psd1'], row['psd2'], row['no_residues'], row['siz'], row['cov'], row['ol'], row['os']]
    e2 = preppi_info[(p1, p2)][1][1]+[row['psd2'], row['psd1'], row['no_residues'], row['siz'], row['cov'], row['ol'], row['os']]

    preppi_info[(p1, p2)] = [e, [e1, e2]]

for interaction in all_interactions:
    p1, p2 = interaction.split('_')

    if p1 == p2:
        structs1 = structures[p1+'_'+p2][p1]
        graphs, gmms = compile_graphs(p1, p2, None, structs1, None, len(fasta_dict[p1]), None, './data/structs', './data/gmm_activity', preppi_info)
        with open('./data/struct_graphs_input/'+p1+'_'+p2+'.pkl', 'wb') as outfile1:
            pickle.dump(graphs, outfile1)
        with open('./data/struct_graphs_gmms_input/'+p1+'_'+p2+'.pkl', 'wb') as outfile2:
            pickle.dump(gmms, outfile2)
    else:
        structs1 = structures[p1+'_'+p2][p1]
        structs2 = structures[p1+'_'+p2][p2]
        graphs, gmms = compile_graphs(p1, p2, 0, structs1, structs2, len(fasta_dict[p1]), len(fasta_dict[p2]), './data/structs', './data/gmm_activity', preppi_info)
        with open('./data/struct_graphs_input/'+p1+'_'+p2+'.pkl', 'wb') as outfile1:
            pickle.dump(graphs, outfile1)
        with open('./data/struct_graphs_gmms_input/'+p1+'_'+p2+'.pkl', 'wb') as outfile2:
            pickle.dump(gmms, outfile2)

        graphs, gmms = compile_graphs(p1, p2, 1, structs1, structs2, len(fasta_dict[p1]), len(fasta_dict[p2]), './data/structs', './data/gmm_activity', preppi_info)
        with open('./data/struct_graphs_input/'+p2+'_'+p1+'.pkl', 'wb') as outfile1:
            pickle.dump(graphs, outfile1)
        with open('./data/struct_graphs_gmms_input/'+p2+'_'+p1+'.pkl', 'wb') as outfile2:
            pickle.dump(gmms, outfile2)

print('done!')
