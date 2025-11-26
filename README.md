# PIONEER2
PIONEER2 is a geometric deep learning-based framework that predicts protein-protein interfaces by integrating structural homologs.

After cloning the repository, please run the following commands for the installation of environment:
```
conda create -n pioneer2 python=3.9
conda activate pioneer2
conda install numpy=1.26.4
conda install pandas=2.2.3
conda install -c pytorch pytorch=1.12.1 torchvision=0.13.1 torchaudio=0.12.1 cudatoolkit=11.3
conda install -c dglteam dgl-cuda11.3
conda install -c conda-forge scikit-learn=0.24.2
conda install -c conda-forge biopython=1.79
conda install -c salilab dssp=3.0.0
conda install -c bioconda cd-hit=4.8.1 msms=2.6.1
pip install pydca
```

## The third-party tools
The following third-party tools are required:\
[makeblastdb](https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/)\
[blastp](https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/)\
[psiblast](https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/)\
[clustalo1.2.4](http://www.clustal.org/omega/)\
[naccess2.1.1](http://www.bioinf.manchester.ac.uk/naccess/)\
[equidock](https://github.com/octavian-ganea/equidock_public/)

## Running PIONEER2
Please enter the folder "pred", and execute the following command as an example:
```
python prediction.py test_interactions.txt result_folder
```
Then you can find the results in the result_folder.
