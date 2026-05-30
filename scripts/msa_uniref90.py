import os, sys

CLUSTALO = './softs/clustalo/clustalo-1.2.4-Ubuntu-x86_64'

def format_rawmsa(prot_id, rawmsa_file, seq_dict, formatted_output_file):
    identifiers_to_align = set()
    with open(rawmsa_file, 'r') as infile:
        for line in infile:
            if line.startswith('>'):
                identifier = line.strip().split()[0]
                if identifier.split('_')[1] != prot_id:
                    identifiers_to_align.add(identifier)
        
    if len(identifiers_to_align) > 0:
        with open(formatted_output_file, 'w') as outfile:
            for identifier in sorted(identifiers_to_align):
                outfile.write(identifier+'\n'+seq_dict[identifier.split('_')[1]]+'\n')
    
def run_clustal(clustal_input_file, clustal_output_file, num_threads=6):
    with open(clustal_input_file, 'r') as f:
        numseqs = len(f.readlines())
    numseqs /= 2
    
    if numseqs > 1:
        os.system('%s -i %s -o %s --force --threads %s' % (CLUSTALO, clustal_input_file, clustal_output_file, str(num_threads)))
    else:
        os.system('cp %s %s' % (clustal_input_file, clustal_output_file))
        
def format_clustal(clustal_output_file, oneline_output_file, formatted_output_file):
    msa_info = []
    with open(clustal_output_file, 'r') as infile:
        seq_name = ''
        seq = ''
        for line in infile:
            if line.startswith('>'):
                if seq_name:
                    msa_info.append(seq_name)
                    msa_info.append(seq)
                seq_name = line.strip()
                seq = ''
            else:
                seq += line.strip()
        msa_info.append(seq_name)
        msa_info.append(seq)
        
    with open(oneline_output_file, 'w') as outfile1:
        for line in msa_info:
            outfile1.write(line+'\n')
            
    outtxt = ''
    gaps = []
    for idx, line in enumerate(msa_info):
        if idx % 2 == 0:
            outtxt += line
            outtxt += '\n'
        elif idx == 1:
            for i in range(len(line)):
                gaps.append(line[i] == '-')
                
        if idx % 2 == 1:
            newseq = ''
            for i in range(len(gaps)):
                if not gaps[i]:
                    if i < len(line):
                        newseq += line[i]
                    else:
                        newseq += '-'
                        
            outtxt += newseq
            outtxt += '\n'
            
    with open(formatted_output_file, 'w') as outfile2:
        outfile2.write(outtxt)

def calculate_msa_uniref90(prot_id, prot_seq, rawmsa_file, seq_dict, output_dir):
    formatted_fasta_file = os.path.join(output_dir, prot_id+'_rawmsa.fasta')
    clustal_input_file = os.path.join(output_dir, prot_id+'.clustal_input')
    clustal_output_file = os.path.join(output_dir, prot_id+'.clustal')
    oneline_clustal_file = os.path.join(output_dir, prot_id+'.oneline_msa')
    formatted_clustal_file = os.path.join(output_dir, prot_id+'.aligned_msa')
    
    format_rawmsa(prot_id, rawmsa_file, seq_dict, formatted_fasta_file)
    
    if os.path.exists(formatted_fasta_file):
        with open(formatted_fasta_file, 'r') as infile:
            lines = infile.readlines()
            
        with open(clustal_input_file, 'w') as outfile:
            outfile.write('>'+prot_id+'\n'+prot_seq+'\n')
            for line in lines:
                outfile.write(line)
                
        run_clustal(clustal_input_file, clustal_output_file)
        
    if os.path.exists(clustal_output_file):
        format_clustal(clustal_output_file, oneline_clustal_file, formatted_clustal_file)
    return formatted_fasta_file, oneline_clustal_file, formatted_clustal_file


fasta_dict = {}
with open('./data/all_preppi_uniprot_seqs.txt', 'r') as infile:
    for line in infile:
        line_list = line.strip().split()
        fasta_dict[line_list[0]] = line_list[1]

seq_dict = {}
with open('./data/uniref90.fasta', 'r') as infile:
    seq = ''
    for line in infile:
        if line.startswith('>'):
            if seq:
                seq_dict[identifier] = seq
            identifier = line.strip().split()[0].split('_')[1]
            seq = ''
        else:
            seq += line.strip()
    seq_dict[identifier] = seq

for identifier in sorted(list(fasta_dict.keys())):
    single_formatted_fasta_file, single_oneline_msa_file, single_aligned_msa_file = calculate_msa_uniref90(identifier, fasta_dict[identifier], './data/psiblast_uniref90/'+identifier+'.rawmsa', seq_dict, './data/msa_uniref90/')

print('done!')
