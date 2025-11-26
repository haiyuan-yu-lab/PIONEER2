import os, sys, json, pickle
import numpy as np
from Bio.PDB import PDBParser

def is_residue(residue):
    try:
        return (residue.get_id()[0] in hetresidue_field) & (residue.resname in residue_dictionary.keys())
    except:
        return False

def get_activity(inputs, centers, sqrt_precision_matrix, eps=0.1, covariance_type='full'):
    center_shape = list(centers.shape)
    if covariance_type == 'diag':
        activity = np.exp(-0.5 * np.sum(
            (
                (np.expand_dims(inputs, axis=-1) 
                 - np.reshape(centers, [1] + center_shape)
                ) / np.reshape(eps+centers, [1] + center_shape)
            )**2, axis=-2))
    elif covariance_type == 'full':
        intermediate = np.expand_dims(inputs, axis=-1) - np.reshape(centers, 
                                                                    [1] + center_shape)
        
        intermediate2 = np.sum(
            np.expand_dims(intermediate, axis=-3) *
            np.expand_dims(sqrt_precision_matrix, axis=0),
            axis=-2)
        
        activity = np.exp(-0.5 * np.sum(intermediate2**2, axis=-2))
    return activity

def make_flat(points):
    d = points.shape[-1]
    points = points.reshape([-1, d])
    return points


with open('./data/structs_to_use.json', 'r') as infile:
    structures = json.load(infile)

all_interactions = sorted(list(structures.keys()))

struct_basefiles = set()
for pp in all_interactions:
    p1, p2 = pp.split('_')
    assert p1 <= p2

    p1_structs = structures[p1+'_'+p2][p1]
    for i, p1_struct in enumerate(p1_structs):
        if p1_struct[0] == 'PDB':
            assert p1_struct[4] != ' '
            struct_basefiles.add(p1+'_'+p1_struct[3]+'_'+p1_struct[4]+'.pdb')
        else:
            assert p1_struct[0] == 'AlphaFold'
            struct_basefiles.add(p1+'_AF.pdb')

    if p1 != p2:
        p2_structs = structures[p1+'_'+p2][p2]
        for i, p2_struct in enumerate(p2_structs):
            if p2_struct[0] == 'PDB':
                assert p2_struct[4] != ' '
                struct_basefiles.add(p2+'_'+p2_struct[3]+'_'+p2_struct[4]+'.pdb')
            else:
                assert p2_struct[0] == 'AlphaFold'
                struct_basefiles.add(p2+'_AF.pdb')

struct_basefiles = sorted(struct_basefiles)

with open('./data/gaussian_kernel_parameters_pdb.pkl', 'rb') as infile:
    gmm_paras = pickle.load(infile)

aa_centers = gmm_paras['aa_centers']
aa_sqrt_precision_matrix = gmm_paras['aa_sqrt_precision_matrix']
atom_centers = gmm_paras['atom_centers']
atom_sqrt_precision_matrix = gmm_paras['atom_sqrt_precision_matrix']

print(len(struct_basefiles))
start = int(sys.argv[1])
end = int(sys.argv[2])
for struct_basefile in struct_basefiles[start:end]:
    print(struct_basefile)

    local_neighborhood_frames_file = './data/local_neighborhood_frames/'+struct_basefile[:-4]+'_local_neighborhood_frames.pkl'

    with open(local_neighborhood_frames_file, 'rb') as infile:
        local_coordinates = pickle.load(infile)

    parser = PDBParser(QUIET=True)
    struct = parser.get_structure('PDB', './data/structs/'+struct_basefile)
    assert len(struct) == 1
    model = struct[0]
    assert len(model) == 1

    pdb_ids = []
    chains = [c for c in model]
    assert len(chains) == 1
    for r in chains[0]:
        ind = r.get_id()
        pdb_ids.append(str(ind[1]).strip()+ind[2].strip())

    assert len(pdb_ids) == local_coordinates['aa_local_coordinates'].shape[0]

    aa_activity = get_activity(make_flat(local_coordinates['aa_local_coordinates']), aa_centers, aa_sqrt_precision_matrix, eps=0.1, covariance_type='full')
    atom_activity = get_activity(make_flat(local_coordinates['atom_local_coordinates']), atom_centers, atom_sqrt_precision_matrix, eps=0.1, covariance_type='full')

    atom_nums = []
    atom_indices = local_coordinates['atom_indices']
    start = 0
    for i in range(1, atom_indices.shape[0]):
        assert len(atom_indices[i]) == 1
        if atom_indices[i][0] != atom_indices[i-1][0]:
            end = i-1
            atom_nums.append([start, end])
            start = i
    atom_nums.append([start, i])
    assert len(atom_nums) == local_coordinates['aa_indices'].shape[0]

    result_file = './data/gmm_activity/'+struct_basefile[:-4]+'.pkl'
    with open(result_file, 'wb') as outfile:
        pickle.dump({'gmm_activities': [aa_activity, atom_activity], 'atom_nums': np.array(atom_nums)}, outfile)
                
print('done!')
