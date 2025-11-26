import numpy as np

aa = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
      'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V',  'W', 'Y', '-']
aadict = {aa[k]: k for k in range(len(aa))}

aadict['X'] = len(aa)
aadict['B'] = len(aa)
aadict['Z'] = len(aa)
aadict['O'] = len(aa)
aadict['U'] = len(aa)

for k, key in enumerate(['a', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'v',  'w', 'y']):
    aadict[key] = aadict[aa[k]]
aadict['x'] = len(aa)
aadict['b'] = len(aa)
aadict['z'] = -1
aadict['.'] = -1

def seq2num(string):
    if type(string) in [str, np.str_]:
        return np.array([aadict[x] for x in string])[np.newaxis, :]
    elif type(string) in [list, np.ndarray]:
        return np.array([[aadict[x] for x in string_] for string_ in string])

list_atoms_types = ['C', 'O', 'N', 'S']
VanDerWaalsRadii = np.array([1.70, 1.52, 1.55, 1.80])

atom_mass = np.array(
    [
        12,
        16,
        14,
        32
    ]
)

atom_type_to_index = dict([(list_atoms_types[i], i)
                           for i in range(len(list_atoms_types))])

list_atoms = ['C', 'CA', 'CB', 'CD', 'CD1', 'CD2', 'CE', 'CE1', 'CE2', 'CE3',
              'CG', 'CG1', 'CG2', 'CH2', 'CZ', 'CZ2', 'CZ3', 'N', 'ND1', 'ND2',
              'NE', 'NE1', 'NE2', 'NH1', 'NH2', 'NZ', 'O', 'OD1', 'OD2', 'OE1',
              'OE2', 'OG', 'OG1', 'OH', 'OXT', 'SD', 'SE', 'SG']

atom_to_index = dict([(list_atoms[i], i) for i in range(len(list_atoms))])
atom_to_index['OT1'] = atom_to_index['O']
atom_to_index['OT2'] = atom_to_index['OXT']

index_to_type = np.zeros(38,dtype=np.int32)
for atom,index in atom_to_index.items():
    index_to_type[index] = list_atoms_types.index(atom[0])

atom_type_mass = np.zeros( 38 )
for atom,index in atom_to_index.items():
    atom_type_mass[index] = atom_mass[index_to_type[index]]

list_aa = [
    'A',
    'C',
    'D',
    'E',
    'F',
    'G',
    'H',
    'I',
    'K',
    'L',
    'M',
    'N',
    'P',
    'Q',
    'R',
    'S',
    'T',
    'V',
    'W',
    'Y'
]

residue_dictionary = {'CYS': 'C', 'CAS': 'C', 'CSS': 'C', 'YCM': 'C', 'CSO': 'C', '2CO': 'C', 'SNC': 'C',
                      'SMC': 'C', 'CSD': 'C', 'SCY': 'C', 'CME': 'C', 'CMH': 'C', '4GJ': 'C', 'XCN': 'C',
                      'ASP': 'D', 'BFD': 'D', 'BHD': 'D', '0TD': 'D', 'SER': 'S', 'GLN': 'Q', 'MGN': 'Q',
                      'LYS': 'K', 'MLY': 'K', 'MLZ': 'K', 'M3L': 'K', 'KCX': 'K', 'ALY': 'K', '5CT': 'K', 'GPL': 'K',
                      'KEO': 'K', 'LLP': 'K',
                      'ILE': 'I', 'PRO': 'P', 'HYP': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N', 'MEN': 'N',
                      'GLY': 'G', 'GLZ': 'G', 'GL3': 'G', 'HIS': 'H', 'MHS': 'H',
                      'LEU': 'L', 'ARG': 'R', 'AGM': 'R', 'TRP': 'W', 'TRX': 'W', 'TTQ': 'W', '0AF': 'W', 'FTR': 'W',
                      'ALA': 'A', 'AYA': 'A', 'VAL': 'V', 'GLU': 'E', 'CGU': 'E', 'GMA': 'E', 'TYR': 'Y', 'MET': 'M',
                      'MSE': 'M', 'CXM': 'M', 'FME': 'M',
                      'PTR':'Y',
                      'TYS':'Y',
                      'SEP':'S','OSE':'S', 'SAC': 'S', 'HSE':'S',
                      'TPO':'T',
                      'HIC':'H', 'HIP':'H', 'NEP':'H',
}

hetresidue_field = [' '] + ['H_%s'%name for name in residue_dictionary.keys()]

aa_to_index = dict([(list_aa[i],i) for i in range(20)])

dictionary_covalent_bonds_ = {
    'A':{
        'C': ['CA', 'N'],
        'CA': ['N', 'C'],
        'N': ['C', 'CA'],
        'O': ['C', None],
        'OXT':['C',None],
        'CB': ['CA', None]
    },
    'C':{
        'C': ['CA', 'N'],
        'CA': ['N', 'C'],
        'N': ['C', 'CA'],
        'O': ['C', None],
        'OXT': ['C', None],
        'CB': ['CA', 'SG'],
        'SG': ['CB', None]
    },
    'D':{
        'C': ['CA', 'N'],
        'CA': ['N', 'C'],
        'N': ['C', 'CA'],
        'O': ['C', None],
        'OXT': ['C', None],
        'CB': ['CA', 'CG'],
        'CG': ['CB', 'OD1'],
        'OD1': ['CG', None],
        'OD2': ['CG', None],
    },
    'E':{
        'C': ['CA', 'N'],
        'CA': ['N', 'C'],
        'N': ['C', 'CA'],
        'O': ['C', None],
        'OXT': ['C', None],
        'CB': ['CA', 'CG'],
        'CG': ['CB', 'CD'],
        'CD': ['CG', 'OE1'],
        'OE1': ['CD', None],
        'OE2': ['CD', None]
    },
    'F':{
        'C': ['CA', 'N'],
        'CA': ['N', 'C'],
        'N': ['C', 'CA'],
        'O': ['C', None],
        'OXT': ['C', None],
        'CB': ['CA', 'CG'],
        'CG': ['CB', 'CD1'],
        'CD1': ['CG', 'CE1'],
        'CE1': ['CD1', 'CZ'],
        'CZ': ['CE1', 'CE2'],
        'CE2': ['CD2', 'CZ'],
        'CD2': ['CG', 'CE2']
    },
    'G':{
        'C': ['CA', 'N'],
        'CA': ['N', 'C'],
        'N': ['C', 'CA'],
        'O': ['C', None],
        'OXT': ['C', None],
    },
    'H':{
        'C': ['CA', 'N'],
        'CA': ['N', 'C'],
        'N': ['C', 'CA'],
        'O': ['C', None],
        'OXT': ['C', None],
        'CB': ['CA', 'CG'],
        'CG': ['CB', 'ND1'],
        'ND1': ['CG', 'CE1'],
        'CE1': ['ND1', 'NE2'],
        'NE2': ['CE1', 'CD2'],
        'CD2': ['NE2', 'CG']
    },
    'I':{
        'C': ['CA', 'N'],
        'CA': ['N', 'C'],
        'N': ['C', 'CA'],
        'O': ['C', None],
        'OXT': ['C', None],
        'CB': ['CA', 'CG1'],
        'CG1': ['CB', 'CD1'],
        'CG2': ['CB', None],
        'CD1':['CG1',None]
    },
    'K':{
        'C': ['CA', 'N'],
        'CA': ['N', 'C'],
        'N': ['C', 'CA'],
        'O': ['C', None],
        'OXT': ['C', None],
        'CB': ['CA', 'CG'],
        'CG': ['CB', 'CD'],
        'CD': ['CG', 'CE'],
        'CE': ['CD', 'NZ'],
        'NZ': ['CE', None],
    },
    'L':{
        'C': ['CA', 'N'],
        'CA': ['N', 'C'],
        'N': ['C', 'CA'],
        'O': ['C', None],
        'OXT': ['C', None],
        'CB': ['CA', 'CG'],
        'CG': ['CB', 'CD1'],
        'CD1': ['CG', None],
        'CD2': ['CG', None]
    },
    'M':{
        'C': ['CA', 'N'],
        'CA': ['N', 'C'],
        'N': ['C', 'CA'],
        'O': ['C', None],
        'OXT': ['C', None],
        'CB': ['CA', 'CG'],
        'CG': ['CB', 'SD'],
        'SD': ['CG', 'CE'],
        'CE':['SD',None]
    },
    'N':{
        'C': ['CA', 'N'],
        'CA': ['N', 'C'],
        'N': ['C', 'CA'],
        'O': ['C', None],
        'OXT': ['C', None],
        'CB': ['CA', 'CG'],
        'CG': ['CB', 'OD1'],
        'OD1': ['CG', None],
        'ND2': ['CG', None]
    },
    'P':{
        'C': ['CA', 'N'],
        'CA': ['N', 'C'],
        'N': ['C', 'CA'],
        'O': ['C', None],
        'OXT': ['C', None],
        'CB': ['CA', 'CG'],
        'CG': ['CB', 'CD'],
        'CD': ['CG', 'N']
    },
    'Q':{
        'C': ['CA', 'N'],
        'CA': ['N', 'C'],
        'N': ['C', 'CA'],
        'O': ['C', None],
        'OXT': ['C', None],
        'CB': ['CA', 'CG'],
        'CG': ['CB', 'CD'],
        'CD': ['CG', 'OE1'],
        'OE1': ['CD', None],
        'NE2': ['CD', None]
    },
    'R':{
        'C': ['CA', 'N'],
        'CA': ['N', 'C'],
        'N': ['C', 'CA'],
        'O': ['C', None],
        'OXT': ['C', None],
        'CB': ['CA', 'CG'],
        'CG': ['CB', 'CD'],
        'CD': ['CG', 'NE'],
        'NE': ['CD', 'CZ'],
        'CZ': ['NE', 'NH1'],
        'NH1': ['CZ', None],
        'NH2': ['CZ', None]
    },
    'S':{
        'C': ['CA', 'N'],
        'CA': ['N', 'C'],
        'N': ['C', 'CA'],
        'O': ['C', None],
        'OXT': ['C', None],
        'CB': ['CA', 'OG'],
        'OG': ['CB', None],
    },
    'T':{
        'C': ['CA', 'N'],
        'CA': ['N', 'C'],
        'N': ['C', 'CA'],
        'O': ['C', None],
        'OXT': ['C', None],
        'CB': ['CA', 'OG1'],
        'OG1': ['CB', None],
        'CG2': ['CB', None]
    },
    'V':{
        'C': ['CA', 'N'],
        'CA': ['N', 'C'],
        'N': ['C', 'CA'],
        'O': ['C', None],
        'OXT': ['C', None],
        'CB': ['CA', 'CG1'],
        'CG1': ['CB', None],
        'CG2': ['CB', None],
    },
    'W':{
        'C': ['CA', 'N'],
        'CA': ['N', 'C'],
        'N': ['C', 'CA'],
        'O': ['C', None],
        'OXT': ['C', None],
        'CB': ['CA', 'CG'],
        'CG': ['CB', 'CD1'],
        'CD1': ['CG', 'NE1'],
        'NE1': ['CD1', 'CE2'],
        'CD2': ['CG', 'CE2'],
        'CE2': ['CD2', 'CZ2'],
        'CZ2': ['CE2', 'CH2'],
        'CH2': ['CZ2', 'CZ3'],
        'CZ3': ['CH2', 'CE3'],
        'CE3': ['CZ3', 'CD2']
    },
    'Y':{
        'C': ['CA', 'N'],
        'CA': ['N', 'C'],
        'N': ['C', 'CA'],
        'O': ['C', None],
        'OXT': ['C', None],
        'CB': ['CA', 'CG'],
        'CG': ['CB', 'CD1'],
        'CD1': ['CG', 'CE1'],
        'CE1': ['CD1', 'CZ'],
        'CZ': ['CE1', 'CE2'],
        'CE2': ['CD2', 'CZ'],
        'CD2': ['CG', 'CE2'],
        'OH': ['CZ', None]
    }
}

dictionary_covalent_bonds = {}
for aa,atom_covalent_bonds in dictionary_covalent_bonds_.items():
    for atom,bonds in atom_covalent_bonds.items():
        bonds_num = -1 * np.ones(3,dtype=np.int32)
        for l,bond in enumerate(bonds):
            if bond is not None:
                bonds_num[l] = atom_to_index[bond]
        dictionary_covalent_bonds['%s_%s'%(aa, atom_to_index[atom] )] = bonds_num

list_atom_valencies = [
    'C',
    'CH',
    'CH2',
    'CH3',
    'CPi',
    'O',
    'OH',
    'N',
    'NH',
    'NH2',
    'S',
    'SH'
]

dictionary_atom_valencies = {
    'A': {
        'C': 'C',
        'CA': 'CH',
        'CB': 'CH3',
        'O': 'O',
        'OXT': 'OH',
        'N': 'NH'
    },
    'C': {
        'C': 'C',
        'CA': 'CH',
        'CB': 'CH2',
        'O': 'O',
        'OXT': 'OH',
        'N': 'NH',
        'SG': 'SH'
    },
    'D': {
        'C': 'C',
        'CA': 'CH',
        'CB': 'CH2',
        'CG': 'C',
        'O': 'O',
        'OD1': 'O',
        'OD2': 'OH',
        'OXT': 'OH',
        'N': 'NH',
    },
    'E': {
        'C': 'C',
        'CA': 'CH',
        'CB': 'CH2',
        'CG': 'CH',
        'CD': 'C',
        'O': 'O',
        'OE1': 'O',
        'OE2': 'OH',
        'OXT': 'OH',
        'N': 'NH',
    },
    'F': {
        'C': 'C',
        'CA': 'CH',
        'CB': 'CH2',
        'CG': 'CPi',
        'CD1': 'CPi',
        'CD2': 'CPi',
        'CE1': 'CPi',
        'CE2': 'CPi',
        'CZ': 'CPi',
        'N': 'NH',
        'O': 'O',
        'OXT': 'OH',
    },
    'G': {
        'C': 'C',
        'CA': 'CH2',
        'N': 'NH',
        'O': 'O',
        'OXT': 'OH'
    },
    'H': {
        'C': 'C',
        'CA': 'CH',
        'CB': 'CH2',
        'CG': 'CPi',
        'CE1': 'CPi',
        'CD2': 'CPi',
        'N': 'NH',
        'ND1': 'N',
        'ND2': 'NH2',
        'O': 'O',
        'OXT': 'OH',
    },
    'I': {
        'C': 'C',
        'CA': 'CH',
        'CB': 'CH',
        'CG1': 'CH2',
        'CG2': 'CH3',
        'CD1': 'CH3',
        'N': 'NH',
        'O': 'O',
        'OXT': 'OH'
    },
    'K': {
        'C': 'C',
        'CA': 'CH',
        'N': 'NH',
        'O': 'O',
        'OXT': 'OH',
        'CB': 'CH2',
        'CG': 'CH2',
        'CD': 'CH2',
        'CE': 'CH2',
        'NZ': 'NH2'
    },
    'L': {
        'C': 'C',
        'CA': 'CH',
        'N': 'NH',
        'O': 'O',
        'OXT': 'OH',
        'CB': 'CH2',
        'CG': 'CH',
        'CD1': 'CH3',
        'CD2': 'CH3',
    },
    'M': {
        'C': 'C',
        'CA': 'CH',
        'N': 'NH',
        'O': 'O',
        'OXT': 'OH',
        'CB': 'CH2',
        'CG': 'CH2',
        'SD': 'S',
        'CE': 'CH3'
    },
    'N': {
        'C': 'C',
        'CA': 'CH',
        'N': 'NH',
        'O': 'O',
        'OXT': 'OH',
        'CB': 'CH2',
        'CG': 'C',
        'OD1': 'O',
        'ND2': 'NH2'
    },
    'P': {
        'C': 'C',
        'CA': 'CH',
        'N': 'N',
        'O': 'O',
        'OXT': 'OH',
        'CB': 'CH2',
        'CG': 'CH2',
        'CD': 'CH2'
    },
    'Q': {
        'C': 'C',
        'CA': 'CH',
        'N': 'NH',
        'O': 'O',
        'OXT': 'OH',
        'CB': 'CH2',
        'CG': 'CH2',
        'CD': 'C',
        'OE1': 'O',
        'NE2': 'NH2'
    },
    'R': {
        'C': 'C',
        'CA': 'CH',
        'N': 'NH',
        'O': 'O',
        'OXT': 'OH',
        'CB': 'CH2',
        'CG': 'CH2',
        'CD': 'CH2',
        'NE': 'NH',
        'CZ': 'C',
        'NH1': 'NH',
        'NH2': 'NH2'
    },
    'S': {
        'C': 'C',
        'CA': 'CH',
        'N': 'NH',
        'O': 'O',
        'OXT': 'OH',
        'CB': 'CH2',
        'OG': 'OH',
    },
    'T': {
        'C': 'C',
        'CA': 'CH',
        'N': 'NH',
        'O': 'O',
        'OXT': 'OH',
        'CB': 'CH',
        'OG1': 'OH',
        'CG2': 'CH3'
    },
    'V': {
        'C': 'C',
        'CA': 'CH',
        'N': 'NH',
        'O': 'O',
        'OXT': 'OH',
        'CB': 'CH',
        'CG1': 'CH3',
        'CG2': 'CH3',
    },
    'W': {
        'C': 'C',
        'CA': 'CH',
        'N': 'NH',
        'O': 'O',
        'OXT': 'OH',
        'CB': 'CH2',
        'CG': 'CPi',
        'CD1': 'CPi',
        'NE1': 'NH',
        'CD2': 'CPi',
        'CE2': 'CPi',
        'CZ2': 'CPi',
        'CH2': 'CPi',
        'CZ3': 'CPi',
        'CE3': 'CPi'
    },
    'Y': {
        'C': 'C',
        'CA': 'CH',
        'N': 'NH',
        'O': 'O',
        'OXT': 'OH',
        'CB': 'CH2',
        'CG': 'CPi',
        'CD1': 'CPi',
        'CE1': 'CPi',
        'CZ': 'CPi',
        'CE2': 'CPi',
        'CD2': 'CPi',
        'OH': 'OH'
    }
}

index_to_valency = np.zeros([20, 38], dtype=np.int32)
for k, aa in enumerate(list_aa):
    for key, value in dictionary_atom_valencies[aa].items():
        i = list_atoms.index(key)
        j = list_atom_valencies.index(value)
        index_to_valency[k, i] = j

expasy = {'HPHO': {
'Ala':  1.800, 'Arg': -4.500, 'Asn': -3.500, 'Asp': -3.500, 'Cys':  2.500, 'Gln': -3.500, 'Glu': -3.500, 'Gly': -0.400, 'His': -3.200, 'Ile':  4.500, 'Leu':  3.800, 'Lys': -3.900, 'Met':  1.900, 'Phe':  2.800, 'Pro': -1.600, 'Ser': -0.800, 'Thr': -0.700, 'Trp': -0.900, 'Tyr': -1.300, 'Val':  4.200,
},

'POLA': {
'Ala':  8.100, 'Arg': 10.500, 'Asn': 11.600, 'Asp': 13.000, 'Cys':  5.500, 'Gln': 10.500, 'Glu': 12.300, 'Gly':  9.000, 'His': 10.400, 'Ile':  5.200, 'Leu':  4.900, 'Lys': 11.300, 'Met':  5.700, 'Phe':  5.200, 'Pro':  8.000, 'Ser':  9.200, 'Thr':  8.600, 'Trp':  5.400, 'Tyr':  6.200, 'Val':  5.900,
},

'AREA': {
'Ala': 86.600, 'Arg': 162.200, 'Asn': 103.300, 'Asp': 97.800, 'Cys': 132.300, 'Gln': 119.200, 'Glu': 113.900, 'Gly': 62.900, 'His': 155.800, 'Ile': 158.000, 'Leu': 164.100, 'Lys': 115.500, 'Met': 172.900, 'Phe': 194.100, 'Pro': 92.900, 'Ser': 85.600, 'Thr': 106.500, 'Trp': 224.600, 'Tyr': 177.700, 'Val': 141.000,
},

'ACCE': {
'Ala':  6.600, 'Arg':  4.500, 'Asn':  6.700, 'Asp':  7.700, 'Cys':  0.900, 'Gln':  5.200, 'Glu':  5.700, 'Gly':  6.700, 'His':  2.500, 'Ile':  2.800, 'Leu':  4.800, 'Lys': 10.300, 'Met':  1.000, 'Phe':  2.400, 'Pro':  4.800, 'Ser':  9.400, 'Thr':  7.000, 'Trp':  1.400, 'Tyr':  5.100, 'Val':  4.500, 
},

'TRAN': {
'Ala':  0.380, 'Arg': -2.570, 'Asn': -1.620, 'Asp': -3.270, 'Cys': -0.300, 'Gln': -1.840, 'Glu': -2.900, 'Gly': -0.190, 'His': -1.440, 'Ile':  1.970, 'Leu':  1.820, 'Lys': -3.460, 'Met':  1.400, 'Phe':  1.980, 'Pro': -1.440, 'Ser': -0.530, 'Thr': -0.320, 'Trp':  1.530, 'Tyr':  0.490, 'Val':  1.460,
},

'BULK': {
'Ala': 11.500, 'Arg': 14.280, 'Asn': 12.820, 'Asp': 11.680, 'Cys': 13.460, 'Gln': 14.450, 'Glu': 13.570, 'Gly': 3.400, 'His': 13.690, 'Ile': 21.400, 'Leu': 21.400, 'Lys': 15.710, 'Met': 16.250, 'Phe': 19.800, 'Pro': 17.430, 'Ser': 9.470, 'Thr': 15.770, 'Trp': 21.670, 'Tyr': 18.030, 'Val': 21.570
},

'COMP': {
'Ala':8.25, 'Arg':5.53, 'Asn':4.06, 'Asp':5.45, 'Cys':1.37, 'Gln':3.93, 'Glu':6.75, 'Gly':7.07, 'His':2.27, 'Ile':5.96, 'Leu':9.66, 'Lys':5.84, 'Met':2.42, 'Phe':3.86, 'Pro':4.70, 'Ser':6.56, 'Thr':5.34, 'Trp':1.08, 'Tyr':2.92, 'Val':6.87
}
}

for prop in list(expasy.keys()):
    for aa in list(expasy[prop].keys()):
        expasy[prop][residue_dictionary[aa.upper()]] = expasy[prop][aa]
