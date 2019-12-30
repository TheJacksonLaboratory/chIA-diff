# coding: utf-8
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

import sys
import logging
from copy import deepcopy
import itertools

sys.path.append('..')
from chia_diff import GenomeLoopData
from chia_diff import ChromLoopData
from chia_diff import util
from chia_diff import chia_diff

MOUSE_DATA_DIR = '/media/hirow/extra/jax/data/chia_pet/mouse'
HUMAN_DATA_DIR = '/media/hirow/extra/jax/data/chia_pet/human'
BEDGRAPH_DATA_DIR = '/media/hirow/extra/jax/data/chia_pet/bedgraphs'
CHIA_DIFF_DIR = '/media/hirow/extra/jax/data/chia_pet/chia_diff'

LOOP_DATA_DIR = '/media/hirow/extra/jax/data/to_use'
PEAK_DATA_DIR = '/media/hirow/extra/jax/data/to_use'
BEDGRAPH_DATA_DIR = '/media/hirow/extra/jax/data/to_use'
BIGWIG_DATA_DIR = '/media/hirow/extra/jax/data/to_use'
CHROM_DATA_DIR = '/media/hirow/extra/jax/data/chrom_sizes'

LOG_MAIN_FORMAT = '%(levelname)s - %(asctime)s - %(name)s:%(filename)s:%(lineno)d - %(message)s'
LOG_BIN_FORMAT = '%(asctime)s - %(filename)s:%(lineno)d\n%(message)s'
LOG_TIME_FORMAT = '%Y-%m-%d %H:%M:%S'

# To see log info statements (optional)
from logging.config import fileConfig
fileConfig('chia_diff.conf')

loop_dict = util.read_data(loop_data_dir=LOOP_DATA_DIR,
                                      chrom_size_file=f'{CHROM_DATA_DIR}/hg38.chrom.sizes',
                                      bigwig_data_dir=BIGWIG_DATA_DIR,
                                      chroms_to_load=['chr1'])


def find_diff():
    print('Starting Function')

    l = deepcopy(loop_dict)
    keys = list(l.keys())

    # Bin Size
    for i in [5000]:

        # Window Size
        for j in [10000000]:

            util.preprocess(l, j)
            for pair in itertools.combinations(keys, 2):
                sample1 = l[pair[0]]
                sample2 = l[pair[1]]
                chia_diff.find_diff_loops(sample1, sample2, bin_size=i,
                                        chroms_to_diff=['chr1'])
                                        # start_index=90718635,
                                        # end_index=92653808)
                                        # start_index=209380000,
                                        # end_index=209722133)
