"""
Use this file for testing purposes in an interactive python session. This is to
avoid the time-consuming process of reloading bedgraphs each time a comparison
is done.
"""

# coding: utf-8
# Automatically updates modules that have been imported
# Still needs to reload/reset if extra methods/fields are added to class objects
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

import sys
import logging
from copy import deepcopy
import os

sys.path.append('..')
from chia_rep import reproducibility
from chia_rep import GenomeLoopData
from chia_rep import ChromLoopData

# MOUSE_DATA_DIR = '/media/hirow/extra/jax/data/chia_pet/prob_mouse'
MOUSE_DATA_DIR = '/media/hirow/extra/jax/data/chia_pet/loops/mouse'
PEAK_DATA_DIR = '/media/hirow/extra/jax/data/chia_pet/peaks'
MY_PEAK_DATA_DIR = '/media/hirow/extra/jax/data/chia_pet/my_peaks'
HUMAN_DATA_DIR = '/media/hirow/extra/jax/data/chia_pet/loops/human'
BIGWIG_DATA_DIR = '/media/hirow/extra/jax/data/chia_pet/bigwigs'
CHROM_DATA_DIR = '/media/hirow/extra/jax/data/chrom_sizes'

LOG_MAIN_FORMAT = '%(levelname)s - %(asctime)s - %(name)s:%(filename)s:%(lineno)d - %(message)s'
LOG_BIN_FORMAT = '%(asctime)s - %(filename)s:%(lineno)d\n%(message)s'
LOG_TIME_FORMAT = '%Y-%m-%d %H:%M:%S'

# To see log info statements (optional)
from logging.config import fileConfig
fileConfig('chia_rep.conf')
main_formatter = logging.Formatter(LOG_MAIN_FORMAT, datefmt=LOG_TIME_FORMAT)

# Make the loop_dict a global variable to be untouched so it doesn't have to be
# reloaded
loop_dict = reproducibility.read_data(loop_data_dir=HUMAN_DATA_DIR,
                                      chrom_size_file=f'{CHROM_DATA_DIR}/hg38.chrom.sizes',
                                      bigwig_data_dir=BIGWIG_DATA_DIR,
                                      peak_data_dir=PEAK_DATA_DIR,
                                      chroms_to_load=['chr1'])

# loop_dict.update(reproducibility.read_data(loop_data_dir=MOUSE_DATA_DIR,
#                                            chrom_size_file=f'{CHROM_DATA_DIR}/mm10.chrom.sizes',
#                                            bigwig_data_dir=BIGWIG_DATA_DIR,
#                                            peak_data_dir=PEAK_DATA_DIR))


# Simply copy/paste following method into interactive session and run it
def comparison():
    parent_dir = 'test_results'
    # parent_dir = 'small_complete'

    directories = [
        parent_dir,
        f'{parent_dir}/results',
        f'{parent_dir}/results/info',
        f'{parent_dir}/tmp'
    ]

    for directory in directories:
        if not os.path.isdir(directory):
            os.mkdir(directory)

    for i in [10]:  # bin size kb
    # for i in [1]:
        i *= 1000
        for j in [10]:  # window size mb
        # for j in [3]:
            j *= 1000000
            for k in [60]:  # Peaks kept
            # for k in [75, 100, 200]:  # Peaks kept
                temp_str = f'half.{k}_peaks.{i}_bin.{j}_window'

                if os.path.isfile(f'{parent_dir}/results/{temp_str}.emd_value.csv'):
                    print(f'Skipping {temp_str}')
                    continue

                main_handler = logging.FileHandler(f'{parent_dir}/tmp/main.{temp_str}.log', mode='w')
                main_handler.setFormatter(main_formatter)
                main_handler.setLevel(logging.DEBUG)

                stream_handler = logging.StreamHandler(sys.stdout)
                stream_handler.setFormatter(main_formatter)
                stream_handler.setLevel(logging.INFO)

                log = logging.getLogger()
                log_all = logging.getLogger('all')

                log.handlers = [main_handler, stream_handler]
                log_all.handlers = [main_handler]

                # Make sure not to modify the original dict
                l = deepcopy(loop_dict)
                reproducibility.preprocess(l, num_peaks=k, kept_dir=f'kept')

                rep, non_rep, emd_scores, j_scores = \
                    reproducibility.compare(l, bin_size=i, window_size=j)
                # reproducibility.output_results(rep, non_rep)
                reproducibility.output_results(rep, non_rep,
                                               f'{parent_dir}/results/info',
                                               temp_str)
                reproducibility.output_to_csv(emd_scores,
                                              f'{parent_dir}/results/{temp_str}.emd_value.csv')
                reproducibility.output_to_csv(j_scores,
                                              f'{parent_dir}/results/{temp_str}.j_value.csv')
