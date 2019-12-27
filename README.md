# ChIA-Rep 
A Python package for assessing the reproducibility of ChIA-PET datasets.    
    
## Methods Overview 
### Preprocessing  
- Keep a loop if one or both of the anchors overlap with a peak, where the peak file may be filtered to keep the top *x* peaks
- Weigh the loop PET count by the anchor intensity
    
### Graph representation and comparison
- For a non-overlapping window, bin the loops into bins of fixed size
- Create an adjacency matrix for each window, where nodes are bins and edges are sums of weighted loop PET counts
- Convert each adjacency matrix into a probability vector by reading row-by-row
- Compute the Jensen-Shannon divergence or the Earth Mover's Distance (EMD) between two probability vectors
- Transform each value to be between -1 (dissimilar) and 1 (similar)
- Take a weighted average of values for all windows in a chromosome
- Genome-wide reproducibility value is computed by averaging over all chromosomes
    
### Example
- Given two ChIA-PET datasets, create adjacency matrices A1 and A2

**Adjacency matrix A1**   

|         | bin1   | bin2   | bin3   | bin4   |
|-------- |------  |------  |------  |------  | 
| bin1    | 3      | 2      | 0      | 1      |
| bin2    |        | 1      | 5      | 3      |
| bin3    |        |        | 10     | 9      |
| bin4    |        |        |        | 20     |
    
**Adjacency matrix A2**   

|         | bin1   | bin2   | bin3   | bin4   |
|-------- |------  |------  |------  |------  | 
| bin1    | 4      | 5      | 1      | 4      |
| bin2    |        | 3      | 2      | 3      |
| bin3    |        |        | 7      | 9      |
| bin4    |        |        |        | 27     |
    
**Probability vectors p_A1 and p_A2**   
- p_A1 = (0.06, 0.05, 0, 0.02, 0.02, 0.009, 0.06, 0.19, 0.17, 0.37)
- p_A2 = (0.06, 0.08, 0.02, 0.06, 0.05, 0.03, 0.05, 0.11, 0.14, 0.42)

## Results
- ChIA-Rep can clearly distinguish between replicates and non-replicates
- Generally, replicates have positive values and non-replicates have negative values
- Can take 0 as a threshold to determine the similarity
    
## Usage 
### Dependencies:
```python
# Automatically included when pip installing
install_requires=[
	'numpy>=1.17.0',  
	'scipy>=1.3.1',  
	'prettytable>=0.7.2',  
	'pybedgraph>=0.5.31',  
	'matplotlib>=3.1.1']
```
### Installation: 
```bash    
git clone https://github.com/c0ver/chia_rep.git    
pip3 install chia_rep/
```
    
### Input data and compare:  
```python    
from chia_rep import reproducibility  
  
TEST_DATA_DIR = 'data'  # Folder containing all data
BIN_SIZE = 10000  # 10kb
WINDOW_SIZE = 10000000  # 10mb
NUM_PEAKS = 60

# To see log info statements (optional)  
from logging.config import fileConfig
fileConfig('chia_rep.conf')
  
  
loop_dict = reproducibility.read_data(loop_data_dir=TEST_DATA_DIR,
                                      chrom_size_file=f'{TEST_DATA_DIR}/hg38.chrom.sizes',
                                      bedgraph_data_dir=TEST_DATA_DIR,
                                      peak_data_dir=TEST_DATA_DIR)
  
reproducibility.preprocess(loop_dict, num_peaks=NUM_PEAKS)
  
rep, non_rep, emd_scores, j_scores = reproducibility.compare(loop_dict, bin_size=BIN_SIZE, window_size=WINDOW_SIZE)  

# To compare only certain samples (further documentation provided in reproducibility.py)
rep, non_rep, emd_scores, j_scores = reproducibility.compare(loop_dict, bin_size=BIN_SIZE, window_size=WINDOW_SIZE, specified_comparisons=[['sampleA1', 'sampleA2'], ['sampleA1', 'sampleB1']])

reproducibility.output_results(rep, non_rep, 'sample_test')  
reproducibility.output_to_csv(emd_scores, 'sample_test/sample_test.emd_value.csv')  
reproducibility.output_to_csv(j_scores, 'sample_test/sample_test.j_value.csv')
```  
  
## Testing  
Use `chia_rep/test/test.py` in an interactive session. Further reasoning provided inside file.

## Contact
Contact Minji (minji.kim@jax.org) for general questions, and report software issues in the [Issues](https://github.com/TheJacksonLaboratory/chia_rep/issues) page. 
