# coding: utf-8
import sys
import logging
from copy import deepcopy
import itertools
import os

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

sys.path.append('..')
from chia_diff import GenomeLoopData
from chia_diff import ChromLoopData
from chia_diff import util
from chia_diff import chia_diff

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

loop_dict = util.read_data(loop_data_dir=LOOP_DATA_DIR,
                           chrom_size_file=f'{CHROM_DATA_DIR}/test.chrom.sizes',
                           bedgraph_data_dir=BEDGRAPH_DATA_DIR,
                           chroms_to_load=['chr1'])


def test_random_walk(bin_size=3):
    print('Testing random walk')

    l = deepcopy(loop_dict)

    if not os.path.isdir('random_walks'):
        os.mkdir('random_walks')

    util.preprocess(l)

    # Bin Size
    for sample in l.values():
        for chrom in sample.chrom_dict.values():
            walks = chia_diff.random_walk(chrom, loop_bin_size=bin_size)

            with open(f'random_walks/{sample.sample_name}.{chrom.name}.'
                      f'bin_size={bin_size}.txt', 'w') as out_file:
                for walk in walks:
                    walk = '\t'.join([str(w) for w in walk])
                    out_file.write(f'{walk}\n')


def find_diff():
    print('Starting Function')

    l = deepcopy(loop_dict)
    keys = list(l.keys())

    # Bin Size
    for i in [3]:

        # Window Size
        for j in [10]:

            util.preprocess(l, j)
            for pair in itertools.combinations(keys, 2):
                sample1 = l[pair[0]]
                sample2 = l[pair[1]]
                chia_diff.find_diff_loops(sample1, sample2, bin_size=i,
                                          chroms_to_diff=['chr1'])
