import os, sys

CLUSTALO = './softs/clustalo/clustalo-1.2.4-Ubuntu-x86_64'

def format_rawmsa(prot_id, rawmsa_file, seq_dict, formatted_output_file):
    identifiers_to_align = set()
    with open(rawmsa_file, 'r') as infile:
        for line in infile:
            if line.startswith('>'):
                identifier = line.strip().split()[0]
                if identifier.split('|')[1] != prot_id:
                    identifiers_to_align.add(identifier)
        
    if len(identifiers_to_align) > 0:
        with open(formatted_output_file, 'w') as outfile:
            for identifier in sorted(identifiers_to_align):
                outfile.write(identifier+'\n'+seq_dict[identifier.split('|')[1]]+'\n')

def run_cdhit(cdhit_input_file, cdhit_output_file):
    os.system('cd-hit -i '+cdhit_input_file+' -o '+cdhit_output_file+' -d 100')
    
def run_clustal(clustal_input_file, clustal_output_file, num_threads=2):
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
    
def parse_cdhit_clstr(cdhit_input_file, cdhit_output_clstr_file):
    msa_dict, identifier2code = {}, {}
    with open(cdhit_input_file, 'r') as infile:
        lines = infile.readlines()
    if len(lines):
        for i in range(0, len(lines), 2):
            msa_dict[lines[i].strip().split('|')[1]] = lines[i+1].strip()
            identifier2code[lines[i].strip().split('|')[1]] = lines[i].strip().split('|')[2].split('_')[-1]
        
    clstr_info = {}
    with open(cdhit_output_clstr_file, 'r') as infile:
        for line in infile:
            if line[0] == '>':
                cluster_id = line.strip()[1:]
                clstr_info[cluster_id] = {}
            else:
                if line.strip()[-1] == '%':
                    clstr_info[cluster_id][line.strip().split('|')[1]] = float(line.strip().split()[-1][:-1])
                else:
                    clstr_info[cluster_id][line.strip().split('|')[1]] = 101
    return msa_dict, identifier2code, clstr_info

def cdhit_clstr_join(prot_id1, prot_id2, cdhit_input_file1, cdhit_input_file2, cdhit_output_clstr_file1, cdhit_output_clstr_file2):
    msa_dict1, identifier2code1, clstr_info1 = parse_cdhit_clstr(cdhit_input_file1, cdhit_output_clstr_file1)
    if prot_id1 != prot_id2:
        msa_dict2, identifier2code2, clstr_info2 = parse_cdhit_clstr(cdhit_input_file2, cdhit_output_clstr_file2)
    else:
        msa_dict2, identifier2code2, clstr_info2 = msa_dict1, identifier2code1, clstr_info1
    
    all_pairs = set(identifier2code1.values()).intersection(set(identifier2code2.values()))
    joined_fasta = {}
    if all_pairs:
        for k1 in sorted(list(clstr_info1.keys())):
            for k2 in sorted(list(clstr_info2.keys())):
                codes1 = [identifier2code1[ele] for ele in clstr_info1[k1]]
                codes2 = [identifier2code2[ele] for ele in clstr_info2[k2]]
                identical_codes = set(codes1).intersection(set(codes2))

                identifiers1 = sorted(clstr_info1[k1].items(), key=lambda x: x[1], reverse=True)
                identifiers2 = sorted(clstr_info2[k2].items(), key=lambda x: x[1], reverse=True)

                for identifier1 in identifiers1:
                    for identifier2 in identifiers2:
                        if identifier2code1[identifier1[0]] == identifier2code2[identifier2[0]] and identifier2code1[identifier1[0]] in all_pairs:
                            all_pairs.remove(identifier2code1[identifier1[0]])
                            joined_fasta[identifier2code1[identifier1[0]]] = msa_dict1[identifier1[0]] + msa_dict2[identifier2[0]]
    return joined_fasta

def calculate_single_msa(prot_id, prot_seq, rawmsa_file, seq_dict, output_dir):
    formatted_fasta_file = os.path.join(output_dir, prot_id+'_rawmsa.fasta')
    cdhit_output_file = os.path.join(output_dir, prot_id+'.cdhit')
    clustal_input_file = os.path.join(output_dir, prot_id+'.clustal_input')
    clustal_output_file = os.path.join(output_dir, prot_id+'.clustal')
    oneline_clustal_file = os.path.join(output_dir, prot_id+'.oneline_msa')
    formatted_clustal_file = os.path.join(output_dir, prot_id+'.aligned_msa')
    
    format_rawmsa(prot_id, rawmsa_file, seq_dict, formatted_fasta_file)
    if os.path.exists(formatted_fasta_file):
        run_cdhit(formatted_fasta_file, cdhit_output_file)
        
    if os.path.exists(cdhit_output_file):
        with open(cdhit_output_file, 'r') as infile:
            lines = infile.readlines()
            
        with open(clustal_input_file, 'w') as outfile:
            outfile.write('>'+prot_id+'\n'+prot_seq+'\n')
            for line in lines:
                outfile.write(line)
                
        run_clustal(clustal_input_file, clustal_output_file)
        
    if os.path.exists(clustal_output_file):
        format_clustal(clustal_output_file, oneline_clustal_file, formatted_clustal_file)
    return formatted_fasta_file, oneline_clustal_file, formatted_clustal_file
        
def calculate_joined_msa(prot_id1, prot_id2, prot_seq1, prot_seq2, cdhit_input_file1, cdhit_input_file2, cdhit_clstr_file1, cdhit_clstr_file2, output_dir):
    clustal_input_file = os.path.join(output_dir, prot_id1+'_'+prot_id2+'.clustal_input')
    clustal_output_file = os.path.join(output_dir, prot_id1+'_'+prot_id2+'.clustal')
    oneline_clustal_file = os.path.join(output_dir, prot_id1+'_'+prot_id2+'.oneline_msa')
    formatted_clustal_file = os.path.join(output_dir, prot_id1+'_'+prot_id2+'.aligned_msa')
    
    joined_fasta = cdhit_clstr_join(prot_id1, prot_id2, cdhit_input_file1, cdhit_input_file2, cdhit_clstr_file1, cdhit_clstr_file2)
    
    if joined_fasta:
        with open(clustal_input_file, 'w') as outfile:
            outfile.write('>'+prot_id1+'_'+prot_id2+'\n')
            outfile.write(prot_seq1+prot_seq2+'\n')

            for k in sorted(joined_fasta.keys()):
                outfile.write('>'+k+'\n')
                outfile.write(joined_fasta[k]+'\n')
                
        run_clustal(clustal_input_file, clustal_output_file)
        
    if os.path.exists(clustal_output_file):
        format_clustal(clustal_output_file, oneline_clustal_file, formatted_clustal_file)
    return oneline_clustal_file, formatted_clustal_file


fasta_dict = {}
with open('./data/all_preppi_uniprot_seqs.txt', 'r') as infile:
    for line in infile:
        line_list = line.strip().split()
        fasta_dict[line_list[0]] = line_list[1]

seq_dict = {}
with open('./data/uniprot_all.fasta', 'r') as infile:
    seq = ''
    for line in infile:
        if line.startswith('>'):
            if seq:
                seq_dict[identifier] = seq
            identifier = line.strip().split('|')[1]
            seq = ''
        else:
            seq += line.strip()
    seq_dict[identifier] = seq

for identifier in sorted(list(fasta_dict.keys())):
    single_formatted_fasta_file, single_oneline_msa_file, single_aligned_msa_file = calculate_single_msa(identifier, fasta_dict[identifier], './data/psiblast/'+identifier+'.rawmsa', seq_dict, './data/msa/')

print('done!')
