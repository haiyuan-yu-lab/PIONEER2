import os, re, gzip, glob, subprocess
from tempfile import mkdtemp
from shutil import rmtree
from urllib.request import urlopen

NACCESS = './softs/naccess2.1.1/naccess'

def unzip_res_range(res_range):
    res_ranges = res_range.strip()[1:-1].split(',')
    index_list = []
    for r in res_ranges:
        if re.match('.+-.+', r):
            a, b = r.split('-')
            index_list += [str(n) for n in range(int(a), int(b)+1)]
        else:
            index_list.append(r)

    if index_list == ['']:
        return []
    else:
        return index_list
    
def naccess(pdb_file):
    cwd = os.getcwd()
    os.chdir(os.path.dirname(pdb_file))

    raw_naccess_output = []
    _, _ = subprocess.Popen([NACCESS, pdb_file, '-h'], stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()
    try:
        raw_naccess_output += open(os.path.splitext(pdb_file)[0]+'.rsa', 'r').readlines()
    except IOError:
        raise IOError('ERROR: Naccess .rsa file was not written. The following command was attempted: %s %s' %(NACCESS, pdb_file))

    os.chdir(cwd)
    return raw_naccess_output
