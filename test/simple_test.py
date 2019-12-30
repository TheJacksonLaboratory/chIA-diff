# coding: utf-8
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

import sys
import logging
from copy import deepcopy
import itertools

sys.path.append('..')
from chia_diff import reproducibility
from chia_diff import GenomeLoopData
from chia_diff import ChromLoopData

LOOP_DATA_DIR = 'test_files'
BEDGRAPH_DATA_DIR = 'test_files'
BIGWIG_DATA_DIR = 'test_files'
CHROM_DATA_DIR = 'test_files'

LOG_MAIN_FORMAT = '%(levelname)s - %(asctime)s - %(name)s:%(filename)s:%(lineno)d - %(message)s'
LOG_BIN_FORMAT = '%(asctime)s - %(filename)s:%(lineno)d\n%(message)s'
LOG_TIME_FORMAT = '%Y-%m-%d %H:%M:%S'

# To see log info statements (optional)
from logging.config import fileConfig
fileConfig('chia_diff.conf')

loop_dict = reproducibility.read_data(loop_data_dir=LOOP_DATA_DIR,
                                      chrom_size_file=f'{CHROM_DATA_DIR}/test.chrom.sizes',
                                      bedgraph_data_dir=BEDGRAPH_DATA_DIR,
                                      chroms_to_load=['chr1'])


def find_diff():
    l = deepcopy(loop_dict)
    keys = list(l.keys())

    # Bin Size
    for i in [3]:

        # Window Size
        for j in [10]:

            reproducibility.preprocess(l, j)
            for pair in itertools.combinations(keys, 2):
                sample1 = l[pair[0]]
                sample2 = l[pair[1]]
                sample1.find_chrom_diff_loops(sample2, bin_size=i,
                                              chroms_to_diff=['chr1'])
