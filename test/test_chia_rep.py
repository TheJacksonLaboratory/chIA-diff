import sys
sys.path.append('..')

from chia_rep import reproducibility
import chia_rep

print(chia_rep.__file__)

LOOP_DATA_DIR = 'test_files'
PEAK_DATA_DIR = 'test_files'
BEDGRAPH_DATA_DIR = 'test_files'
BIGWIG_DATA_DIR = 'test_files'
CHROM_DATA_DIR = 'test_files'

loop_dict = reproducibility.read_data(loop_data_dir=LOOP_DATA_DIR,
                                      chrom_size_file=f'{CHROM_DATA_DIR}/hg38.chrom.sizes',
                                      bedgraph_data_dir=BEDGRAPH_DATA_DIR,
                                      peak_data_dir=PEAK_DATA_DIR,
                                      chroms_to_load=['chr1'])

reproducibility.preprocess(loop_dict, num_peaks=60)

rep, non_rep, emd_scores, j_scores = \
    reproducibility.compare(loop_dict, bin_size=10000, window_size=10000000)

assert str(rep) == '{}'

assert str(non_rep) == "{'sampleA2_sampleB1': {'emd_value': -0.9694930056882819, 'j_value': -0.9904915864194117}, 'sampleA2_sampleA1': {'emd_value': -0.19155216340755665, 'j_value': -0.1451783503725695}, 'sampleB1_sampleA1': {'emd_value': -0.9830894195601206, 'j_value': -0.9946288165331302}}"
assert str(emd_scores) == "OrderedDict([('sampleA2', OrderedDict([('sampleA2', 1), ('Sample Name', 'sampleA2'), ('sampleB1', -0.9904915864194117), ('sampleA1', -0.1451783503725695)])), ('sampleB1', OrderedDict([('sampleB1', 1), ('Sample Name', 'sampleB1'), ('sampleA2', -0.9904915864194117), ('sampleA1', -0.9946288165331302)])), ('sampleA1', OrderedDict([('sampleA1', 1), ('Sample Name', 'sampleA1'), ('sampleA2', -0.1451783503725695), ('sampleB1', -0.9946288165331302)]))])"
assert str(j_scores) == "OrderedDict([('sampleA2', OrderedDict([('sampleA2', 1), ('Sample Name', 'sampleA2')])), ('sampleB1', OrderedDict([('sampleB1', 1), ('Sample Name', 'sampleB1')])), ('sampleA1', OrderedDict([('sampleA1', 1), ('Sample Name', 'sampleA1')]))])"

print("Passed all tests!")
