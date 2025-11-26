import os, sys, json, pickle, copy
import numpy as np
from Bio.PDB import PDBParser, Selection
from protein_physical_chemistry import list_atoms, list_atoms_types, VanDerWaalsRadii, atom_mass, atom_type_to_index, atom_to_index, index_to_type, index_to_valency, atom_type_mass, residue_dictionary, hetresidue_field, dictionary_covalent_bonds, aa_to_index, seq2num

def is_residue(residue):
    try:
        return (residue.get_id()[0] in hetresidue_field) & (residue.resname in residue_dictionary.keys())
    except:
        return False

def is_heavy_atom(atom):
    try:
        return (atom.get_id() in atom_to_index.keys() )
    except:
        return False

def binarize_categorical(matrix, n_classes, out=None):
    L = matrix.shape[0]
    matrix = matrix.astype(np.int32)
    if out is None:
        out = np.zeros([L, n_classes], dtype=np.bool_)
    subset = (matrix>=0) & (matrix<n_classes)
    out[np.arange(L)[subset],matrix[subset]] = 1
    return out

def remove_nan(matrix, value=0.):
    aa_has_nan = np.isnan(matrix).reshape([len(matrix),-1]).max(-1)
    matrix[aa_has_nan] = value
    return matrix

def process_chain(chain):
    sequence = ''
    backbone_coordinates = []
    all_coordinates = []
    all_atoms = []
    all_atom_types = []
    for residue in Selection.unfold_entities(chain, 'R'):
        if is_residue(residue):
            sequence += residue_dictionary[residue.resname]
            residue_atom_coordinates = np.array([atom.get_coord() for atom in residue if is_heavy_atom(atom)])
            residue_atoms = [atom_to_index[atom.get_id()] for atom in residue if is_heavy_atom(atom)]
            residue_atom_type = [atom_type_to_index[atom.get_id()[0]] for atom in residue if is_heavy_atom(atom)]
            residue_backbone_coordinates = []
            for atom in ['N', 'C', 'CA', 'O', 'CB']:
                try:
                    residue_backbone_coordinates.append(residue_atom_coordinates[residue_atoms.index(atom_to_index[atom])])
                except:
                    residue_backbone_coordinates.append(np.ones(3, dtype=np.float32) * np.nan)
            backbone_coordinates.append(residue_backbone_coordinates)
            all_coordinates.append(residue_atom_coordinates)
            all_atoms.append(residue_atoms)
            all_atom_types.append(residue_atom_type)
    backbone_coordinates = np.array(backbone_coordinates)
    return sequence, backbone_coordinates, all_coordinates, all_atoms, all_atom_types

def get_aa_frameCloud_triplet_sidechain(atom_coordinates, atom_ids, verbose=True):
    L = len(atom_coordinates)
    aa_clouds = []
    aa_triplets = []
    count = 0
    for l in range(L):
        atom_coordinate = atom_coordinates[l]
        atom_id = atom_ids[l]
        natoms = len(atom_id)
        if 1 in atom_id:
            calpha_coordinate = atom_coordinate[atom_id.index(1)]
        else:
            if verbose:
                print('Warning, pathological amino acid missing calpha', l)
            calpha_coordinate = atom_coordinate[0]

        center = 1 * count
        aa_clouds.append(calpha_coordinate)
        count += 1
        if count > 1:
            previous = aa_triplets[-1][0]
        else:
            virtual_calpha_coordinate = 2 * calpha_coordinate - atom_coordinates[1][0]
            aa_clouds.append(virtual_calpha_coordinate)
            previous = 1 * count
            count += 1

        sidechain_CoM = np.zeros(3, dtype=np.float32)
        sidechain_mass = 0.
        for n in range(natoms):
            if not atom_id[n] in [0, 1, 17, 26, 34]:
                mass = atom_type_mass[atom_id[n]]
                sidechain_CoM += mass * atom_coordinate[n]
                sidechain_mass += mass
        if sidechain_mass > 0:
            sidechain_CoM /= sidechain_mass
        else:
            if l>0:
                if (0 in atom_id) & (1 in atom_id) & (17 in atom_ids[l-1]):
                    sidechain_CoM = 3 * atom_coordinate[atom_id.index(1)] - atom_coordinates[l-1][atom_ids[l-1].index(17)] - \
                                    atom_coordinate[atom_id.index(0)]
                else:
                    if verbose:
                        print('Warning, pathological amino acid missing side chain and backbone', l)
                    sidechain_CoM = atom_coordinate[-1]
            else:
                if verbose:
                    print('Warning, pathological amino acid missing side chain and backbone', l)
                sidechain_CoM = atom_coordinate[-1]

        aa_clouds.append(sidechain_CoM)
        next = 1 * count
        count += 1
        aa_triplets.append((center, previous, next))
    return aa_clouds, aa_triplets

def get_aa_frameCloud(atom_coordinates, atom_ids, verbose=True):
    aa_clouds, aa_triplets = get_aa_frameCloud_triplet_sidechain(atom_coordinates, atom_ids, verbose=verbose)
    aa_indices = np.arange(len(atom_coordinates)).astype(np.int32)[:, np.newaxis]
    aa_clouds = np.array(aa_clouds)
    aa_triplets = np.array(aa_triplets, dtype=np.int32)
    return aa_clouds, aa_triplets, aa_indices

def get_atom_triplets(sequence, atom_ids, dictionary_covalent_bonds):
    L = len(sequence)
    atom_triplets = []
    all_keys = list(dictionary_covalent_bonds.keys() )
    current_natoms = 0
    for l in range(L):
        aa = sequence[l]
        atom_id = atom_ids[l]
        natoms = len(atom_id)
        for n in range(natoms):
            id = atom_id[n]
            if (id == 17):
                if l > 0:
                    if 0 in atom_ids[l - 1]:
                        previous = current_natoms - len(atom_ids[l - 1]) + atom_ids[l - 1].index(0)
                    else:
                        previous = -1
                else:
                    previous = -1
                if 1 in atom_id:
                    next = current_natoms + atom_id.index(1)
                else:
                    next = -1
            elif (id == 0):
                if 1 in atom_id:
                    previous = current_natoms + atom_id.index(1)
                else:
                    previous = -1
                if l < L - 1:
                    if 17 in atom_ids[l + 1]:
                        next = current_natoms + natoms + atom_ids[l + 1].index(17)
                    else:
                        next = -1
                else:
                    next = -1
            else:
                key = (aa + '_' + str(id) )
                if key in all_keys:
                    previous_id, next_id, _ = dictionary_covalent_bonds[(aa + '_' + str(id) )]
                else:
                    print('Strange atom', (aa + '_' + str(id) ))
                    previous_id = -1
                    next_id = -1
                if previous_id in atom_id:
                    previous = current_natoms + atom_id.index(previous_id)
                else:
                    previous = -1
                if next_id in atom_id:
                    next = current_natoms + atom_id.index(next_id)
                else:
                    next = -1
            atom_triplets.append((current_natoms + n, previous, next))
        current_natoms += natoms
    return atom_triplets

def _add_virtual_atoms(atom_clouds, atom_triplets, verbose=True):
    natoms = len(atom_triplets)
    virtual_atom_clouds = []
    count_virtual_atoms = 0
    centers = list(atom_triplets[:, 0])
    atom_triplets_updated = copy.deepcopy(atom_triplets)
    for n in range(natoms):
        triplet = atom_triplets_updated[n]
        case1 = (triplet[1] >= 0) & (triplet[2] >= 0)
        case2 = (triplet[1] < 0) & (triplet[2] >= 0)
        case3 = (triplet[1] >= 0) & (triplet[2] < 0)
        case4 = (triplet[1] < 0) & (triplet[2] < 0)
        if case1:
            continue
            
        elif case2:
            next_triplet = atom_triplets[centers.index(triplet[2])]
            if next_triplet[2] >= 0:
                virtual_atom = atom_clouds[next_triplet[0]] - atom_clouds[next_triplet[2]] + atom_clouds[triplet[0]]
            else:
                if verbose:
                    print('Pathological case, atom has only one bond and its next partner too', triplet[0], triplet[2])
                virtual_atom = atom_clouds[triplet[0]] + np.array([1, 0, 0])
            virtual_atom_clouds.append(virtual_atom)
            triplet[1] = natoms + count_virtual_atoms
            count_virtual_atoms += 1
            
        elif case3:
            previous_triplet = atom_triplets[centers.index(triplet[1])]
            if previous_triplet[1] >= 0:
                virtual_atom = atom_clouds[previous_triplet[0]] - atom_clouds[previous_triplet[1]] + atom_clouds[
                    triplet[0]]
            else:
                if verbose:
                    print('Pathological case, atom has only one bond and its previous partner too', triplet[0],
                          triplet[1])
                virtual_atom = atom_clouds[triplet[0]] + np.array([0, 0, 1])
            virtual_atom_clouds.append(virtual_atom)
            triplet[2] = natoms + count_virtual_atoms
            count_virtual_atoms += 1
            
        elif case4:
            if verbose:
                print('Pathological case, atom has no bonds at all', triplet[0])
            virtual_previous_atom = atom_clouds[triplet[0]] + np.array([1, 0, 0])
            virtual_next_atom = atom_clouds[triplet[0]] + np.array([0, 0, 1])
            virtual_atom_clouds.append(virtual_previous_atom)
            virtual_atom_clouds.append(virtual_next_atom)
            triplet[1] = natoms + count_virtual_atoms
            triplet[2] = natoms + count_virtual_atoms + 1
            count_virtual_atoms += 2
    return virtual_atom_clouds, atom_triplets_updated

def add_virtual_atoms(atom_clouds, atom_triplets, verbose=True):
    virtual_atom_clouds, atom_triplets = _add_virtual_atoms(atom_clouds, atom_triplets, verbose=verbose)
    if len(virtual_atom_clouds) > 0:
        virtual_atom_clouds = np.array(virtual_atom_clouds)
        if np.abs(virtual_atom_clouds).max() >1e8:
            print('The weird bug happened again at add_virtual_atoms, need to fix virtual atoms')
            weird_indices = np.nonzero(np.abs(virtual_atom_clouds).max(-1) >1e8 )[0]
            print('Fixing %s virtual atoms'%len(weird_indices))
            original_atom_indices = np.array([np.nonzero((atom_triplets[:,1:] == len(atom_triplets)+ index).max(-1))[0][0] for index in weird_indices])
            for weird_index, original_atom_index in zip(weird_indices,original_atom_indices):
                virtual_atom_clouds[weird_index] = atom_clouds[original_atom_index,:]
                if atom_triplets[original_atom_index,1] == weird_index:
                    virtual_atom_clouds[weird_index][0] +=1
                else:
                    virtual_atom_clouds[weird_index][2] += 1
        atom_clouds = np.concatenate([atom_clouds, np.array(virtual_atom_clouds)], axis=0)
    return atom_clouds, atom_triplets

def get_atom_frameCloud(sequence, atom_coordinates, atom_ids):
    atom_clouds = np.concatenate(atom_coordinates, axis=0)
    atom_attributes = np.concatenate(atom_ids, axis=-1)
    atom_triplets = np.array(get_atom_triplets(sequence, atom_ids, dictionary_covalent_bonds),
                             dtype=np.int32)
    atom_indices = np.concatenate([np.ones(len(atom_ids[l]), dtype=np.int32) * l for l in range(len(sequence))],
                                  axis=-1)[:, np.newaxis]
    return atom_clouds, atom_triplets, atom_attributes, atom_indices

def build_dataset(chain):
    sequence, backbone_coordinates, atomic_coordinates, atom_ids, atom_types = process_chain(chain)
    
    aa_clouds, aa_triplets, aa_indices = get_aa_frameCloud(atomic_coordinates, atom_ids, verbose=True)
    aa_attributes = binarize_categorical(seq2num(sequence)[0], 20).astype(np.float32)
    aa_clouds = remove_nan(aa_clouds, value=0.)
    aa_triplets = remove_nan(aa_triplets, value=-1)
    aa_attributes = remove_nan(aa_attributes, value=0.)
    aa_indices = remove_nan(aa_indices, value=-1 )
    
    atom_clouds, atom_triplets, atom_attributes, atom_indices = get_atom_frameCloud(sequence, atomic_coordinates, atom_ids)
    atom_clouds, atom_triplets = add_virtual_atoms(atom_clouds, atom_triplets, verbose=True)
    atom_attributes = index_to_valency[seq2num(sequence)[0,atom_indices[:,0]], atom_attributes]
    atom_clouds = remove_nan(atom_clouds, value=0.)
    atom_attributes = remove_nan(atom_attributes, value=-1)
    atom_triplets = remove_nan(atom_triplets, value=-1)
    atom_indices = remove_nan(atom_indices, value=-1)
    atom_attributes += 1
    
    inputs = {}
    inputs['aa_triplets'] = aa_triplets
    inputs['aa_attributes'] = aa_attributes
    inputs['aa_indices'] = aa_indices
    inputs['aa_clouds'] = aa_clouds
    inputs['atom_triplets'] = atom_triplets
    inputs['atom_attributes'] = atom_attributes
    inputs['atom_indices'] = atom_indices
    inputs['atom_clouds'] = atom_clouds
    return inputs

def get_frames(inputs, epsilon=1e-6):
    triplets, points = inputs[0], inputs[1]
    
    xaxis_ = np.array([[1, 0, 0]], dtype=np.float32)
    yaxis_ = np.array([[0, 1, 0]], dtype=np.float32)
    zaxis_ = np.array([[0, 0, 1]], dtype=np.float32)
    
    triplets = np.clip(triplets, 0, points.shape[-2]-1)
    
    centers = points[triplets[:, 0], :]
    delta_10 = points[triplets[:, 2], :] - points[triplets[:, 0], :]
    delta_20 = points[triplets[:, 1], :] - points[triplets[:, 0], :]
    
    zaxis = (delta_10 + epsilon * zaxis_) / (np.sqrt(np.sum(np.power(delta_10, 2), axis=-1, keepdims=True)) + epsilon)
    
    yaxis = np.cross(zaxis, delta_20)
    yaxis = (yaxis + epsilon * yaxis_) / (np.sqrt(np.sum(np.power(yaxis, 2), axis=-1, keepdims=True)) + epsilon)
    
    xaxis = np.cross(yaxis, zaxis)
    xaxis = (xaxis + epsilon * xaxis_) / (np.sqrt(np.sum(np.power(xaxis, 2), axis=-1, keepdims=True)) + epsilon)
    
    frames = np.stack((centers,xaxis,yaxis,zaxis), axis=-2)
    return frames

def distance(coordinates1, coordinates2, squared=False, ndims=3):
    D = (np.expand_dims(coordinates1[...,0],axis=-1) - np.expand_dims(coordinates2[...,0],axis=-2) )**2
    for n in range(1,ndims):
        D += (np.expand_dims(coordinates1[..., n], axis=-1) - np.expand_dims(coordinates2[..., n], axis=-2)) ** 2
    if not squared:
        D = np.sqrt(D)
    return D

def calculate_LocalCoordinates(frames, Kmax=16):
    first_centers, second_centers = frames[:,0], frames[:,0]
    distance_square = distance(first_centers, second_centers, squared=True,ndims=3)
    
    neighbors = np.expand_dims(np.argsort(distance_square)[:, :Kmax], axis=-1)
    
    local_coordinates = np.sum(np.expand_dims(
        second_centers[neighbors[:,:,0]]
        - np.expand_dims(first_centers, axis=-2),
        axis=-2)
        * np.expand_dims(frames[:,1:4], axis=-3)
        , axis=-1)
    return local_coordinates

def claculate_LocalNeighborhood(pdb_file, Kmax=16, Dmax=None):
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure('PDB', pdb_file)
    model = struct[0]
    chains = [c for c in model]
    assert len(chains) == 1
            
    inputs = build_dataset(chains[0])
    aa_frames = get_frames([inputs['aa_triplets'], inputs['aa_clouds']])
    atom_frames = get_frames([inputs['atom_triplets'], inputs['atom_clouds']])

    aa_local_coordinates = calculate_LocalCoordinates(aa_frames, Kmax=16)
    atom_local_coordinates = calculate_LocalCoordinates(atom_frames, Kmax=16)
        
    data = {'aa_local_coordinates':aa_local_coordinates, 'atom_local_coordinates':atom_local_coordinates}
    data.update(inputs)
    return data


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
for struct_basefile in struct_basefiles:
    data = claculate_LocalNeighborhood('./data/structs/'+struct_basefile, Kmax=16)
    with open('./data/local_neighborhood_frames/'+struct_basefile[:-4]+'_local_neighborhood_frames.pkl', 'wb') as outfile:
        pickle.dump(data, outfile)

print('done!')
