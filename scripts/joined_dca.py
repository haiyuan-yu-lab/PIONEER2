import os, sys

def calculate_joined_dca(input_file, plmdca_output_dir, mfdca_output_dir, num_threads=8):
    os.system('plmdca compute_fn protein %s --max_iterations 500 --num_threads %s --output_dir %s --apc' % (input_file, str(num_threads), plmdca_output_dir))
    os.system('mfdca compute_fn protein %s --output_dir %s --apc' % (input_file, mfdca_output_dir))


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
    p1, p2 = interaction[0], interaction[1]

    input_file = './data/joined_msa/'+p1+'_'+p2+'.aligned_msa'
    if os.path.exists(input_file):
        calculate_joined_dca(input_file, './data/joined_raw_plmdca', './data/joined_raw_mfdca', 8)
    else:
        pass

print('done!')
