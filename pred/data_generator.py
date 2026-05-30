import os, pickle, hashlib, random, torch, dgl
from scipy import sparse as sp
import numpy as np
    
def laplacian_positional_encoding(g, pos_enc_dim):
    A = g.adjacency_matrix(scipy_fmt='csr').astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR', tol=1e-2)
    EigVec = EigVec[:, EigVal.argsort()]
    g.ndata['lap_pos_enc'] = torch.from_numpy(EigVec[:,1:pos_enc_dim+1].real).float()
    return g

def wl_positional_encoding(g):
    max_iter = 2
    node_color_dict = {}
    node_neighbor_dict = {}

    edge_list = torch.nonzero(g.adj().to_dense() != 0, as_tuple=False).numpy()
    node_list = g.nodes().numpy()
    
    for node in node_list:
        node_color_dict[node] = 1
        node_neighbor_dict[node] = {}

    for pair in edge_list:
        u1, u2 = pair
        if u1 not in node_neighbor_dict:
            node_neighbor_dict[u1] = {}
        if u2 not in node_neighbor_dict:
            node_neighbor_dict[u2] = {}
        node_neighbor_dict[u1][u2] = 1
        node_neighbor_dict[u2][u1] = 1
        
    iteration_count = 1
    exit_flag = False
    while not exit_flag:
        new_color_dict = {}
        for node in node_list:
            neighbors = node_neighbor_dict[node]
            neighbor_color_list = [node_color_dict[neb] for neb in neighbors]
            color_string_list = [str(node_color_dict[node])] + sorted([str(color) for color in neighbor_color_list])
            color_string = "_".join(color_string_list)
            hash_object = hashlib.md5(color_string.encode())
            hashing = hash_object.hexdigest()
            new_color_dict[node] = hashing
        color_index_dict = {k: v+1 for v, k in enumerate(sorted(set(new_color_dict.values())))}
        for node in new_color_dict:
            new_color_dict[node] = color_index_dict[new_color_dict[node]]
        if node_color_dict == new_color_dict or iteration_count == max_iter:
            exit_flag = True
        else:
            node_color_dict = new_color_dict
        iteration_count += 1
        
    g.ndata['wl_pos_enc'] = torch.LongTensor(list(node_color_dict.values()))
    return g

def process_data(g, x_col_mean, e_col_mean, x_min, x_max, e_min, e_max):
    if torch.isnan(g.ndata['feat']).any():
        inds = torch.where(torch.isnan(g.ndata['feat']))
        g.ndata['feat'][inds] = torch.take(x_col_mean, inds[1])
    g.ndata['feat'] = (g.ndata['feat']-x_min)/(x_max-x_min)
    g.ndata['feat'][g.ndata['feat']>1] = 1.0
    g.ndata['feat'][g.ndata['feat']<0] = 0.0

    if torch.isnan(g.edata['feat']).any():
        inds = torch.where(torch.isnan(g.edata['feat']))
        g.edata['feat'][inds] = torch.take(e_col_mean, inds[1])
    g.edata['feat'] = (g.edata['feat']-e_min)/(e_max-e_min)
    g.edata['feat'][g.edata['feat']>1] = 1.0
    g.edata['feat'][g.edata['feat']<0] = 0.0
    return g

def process_gmm(file, frame_aa_Kmax, frame_atom_Kmax):
    target_aa_gmms, target_atom_gmms, partner_aa_gmms, partner_atom_gmms = [], [], [], []
    with open(file, 'rb') as infile:
        gmm_info = pickle.load(infile)

    for ele in gmm_info[0]:
        for i in range(0, ele.size()[0], frame_aa_Kmax):
            target_aa_gmms.append(torch.mean(ele[i:i+frame_aa_Kmax, :], 0, keepdim=True))

    for ele in gmm_info[1]:
        for i in range(0, ele.size()[0], frame_atom_Kmax):
            target_atom_gmms.append(torch.mean(ele[i:i+frame_atom_Kmax, :], 0, keepdim=True))

    target_atom_nums = gmm_info[2]

    for ele in gmm_info[3]:
        for i in range(0, ele.size()[0], frame_aa_Kmax):
            partner_aa_gmms.append(torch.mean(ele[i:i+frame_aa_Kmax, :], 0, keepdim=True))

    for ele in gmm_info[4]:
        for i in range(0, ele.size()[0], frame_atom_Kmax):
            partner_atom_gmms.append(torch.mean(ele[i:i+frame_atom_Kmax, :], 0, keepdim=True))

    partner_atom_nums = gmm_info[5]

    target_aa_gmms = torch.cat(target_aa_gmms, 0)
    target_atom_gmms = torch.cat(target_atom_gmms, 0)
    partner_aa_gmms = torch.cat(partner_aa_gmms, 0)
    partner_atom_gmms = torch.cat(partner_atom_gmms, 0)
    return [target_aa_gmms, target_atom_gmms, target_atom_nums, partner_aa_gmms, partner_atom_gmms, partner_atom_nums]

def load_dataset(interactions, batch_size, graph_feature_dir, gmm_feature_dir, net_params, mean_values, min_values, max_values, frame_aa_Kmax, frame_atom_Kmax, shuffle):
    target_x_col_mean, target_e_col_mean, partner_x_col_mean, partner_e_col_mean = mean_values
    target_x_min, target_e_min, partner_x_min, partner_e_min = min_values
    target_x_max, target_e_max, partner_x_max, partner_e_max = max_values

    data = []
    batch_data = []
    if shuffle:
        random.shuffle(interactions)
    for interaction in interactions:
        p1, p2 = interaction
        assert p1 <= p2

        with open(os.path.join(graph_feature_dir, p1+'_'+p2+'.pkl'), 'rb') as infile:
            graphs = pickle.load(infile)

        graphs[0] = process_data(graphs[0], target_x_col_mean, target_e_col_mean, target_x_min, target_x_max, target_e_min, target_e_max)
        graphs[3] = process_data(graphs[3], partner_x_col_mean, partner_e_col_mean, partner_x_min, partner_x_max, partner_e_min, partner_e_max)
        
        graphs += process_gmm(os.path.join(gmm_feature_dir, p1+'_'+p2+'.pkl'), frame_aa_Kmax, frame_atom_Kmax)
        graphs.append(p1+'_'+p2)
        batch_data.append(graphs)
        
        if len(batch_data) >= batch_size:
            data.append(batch_data)
            batch_data = []

        if p1 != p2:
            with open(os.path.join(graph_feature_dir, p2+'_'+p1+'.pkl'), 'rb') as infile:
                graphs = pickle.load(infile)

            graphs[0] = process_data(graphs[0], target_x_col_mean, target_e_col_mean, target_x_min, target_x_max, target_e_min, target_e_max)
            graphs[3] = process_data(graphs[3], partner_x_col_mean, partner_e_col_mean, partner_x_min, partner_x_max, partner_e_min, partner_e_max)
            
            graphs += process_gmm(os.path.join(gmm_feature_dir, p2+'_'+p1+'.pkl'), frame_aa_Kmax, frame_atom_Kmax)
            graphs.append(p2+'_'+p1)
            batch_data.append(graphs)
            
            if len(batch_data) >= batch_size:
                data.append(batch_data)
                batch_data = []
            
    if batch_data:
        assert len(batch_data) < batch_size
        data.append(batch_data)

    for i in range(len(data)):
        for j in range(len(data[i])):
            assert not torch.isnan(data[i][j][0].ndata['feat']).any()
            assert not torch.isnan(data[i][j][0].edata['feat']).any()
            assert not torch.isnan(data[i][j][3].ndata['feat']).any()
            assert not torch.isnan(data[i][j][3].edata['feat']).any()

    if net_params['lap_pos_enc']:
        for i in range(len(data)):
            for j in range(len(data[i])):
                data[i][j][0] = laplacian_positional_encoding(data[i][j][0], net_params['pos_enc_dim'])
                data[i][j][3] = laplacian_positional_encoding(data[i][j][3], net_params['pos_enc_dim'])

    if net_params['wl_pos_enc']:
        for i in range(len(data)):
            for j in range(len(data[i])):
                data[i][j][0] = wl_positional_encoding(data[i][j][0])
                data[i][j][3] = wl_positional_encoding(data[i][j][3])
    return data
