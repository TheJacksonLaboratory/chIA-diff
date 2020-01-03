import os
from collections import OrderedDict
import logging
from pyBedGraph import BedGraph
from .GenomeLoopData import GenomeLoopData

log = logging.getLogger()

VERSION = 11

REPLICATES = [
    ['LHM0011_0011H', 'LHM0014_0014H'],
    ['LHM0009_0009H', 'LHM0013_0013H'],
    ['LMP0002_0002V', 'LMP0001_0001V'],
    ['LHH0083_0083H', 'LHH0085_0085V'],
    ['LMP0004_0004V', 'LMP0003_0003V'],
    ['LHM0007_0007H', 'LHM0010H'],
    ['LHM0008_0008H', 'LHM0012_0012H'],
    ['LME0034_0034V', 'LME0028_0028N'],
    ['LHH0084_0084H', 'LHH0086_0086V'],
    ['LHH0048_0048H', 'LHH0054L_0054H'],
    ['LHH0061_0061H', 'LHH0058_0058H']
]

# Non-replicates that had high reproducibility values (> -0.5)
NON_REP_PROB = [
    'LMP0002_0002V', 'LME0033_0033V',
    'LMP0001_0001V', 'LME0033_0033V',
    'LMP0003_0003V', 'LME0028_0028N'
]

TO_CHECK = REPLICATES + NON_REP_PROB


def read_data(loop_data_dir, chrom_size_file, bigwig_data_dir=None,
              bedgraph_data_dir=None, min_loop_value=0, min_bedgraph_value=0,
              chroms_to_load=None):
    """
    Reads all samples that are found in loop_data_dir.

    loop_data_dir/peak_data_dir/bedgraph_data_dir do not have to be separate
    directories.

    Parameters
    ----------
    loop_data_dir
        Directory with loop files. File names must end in either ".BE3" or
        ".cis"
    chrom_size_file
        Path to chromosome size file
    min_loop_value
        Minimum loop value accepted by GenomeLoopData/ChromLoopData
    bigwig_data_dir : str, optional
        Directory with bigwig files. File names must include "bigwig" case
        insensitive.
    bedgraph_data_dir : str, optional
        Directory with bedgraph files. File names must include "bedgraph" case
        insensitive.
    min_bedgraph_value : int, optional
        Minimum value accepted by BedGraph obj from pyBedGraph
    chroms_to_load : list, optional
        Specify specific chromosomes to load instead of the entire genome

    Returns
    -------
    OrderedDict(str, GenomeLoopData)
        Key: Name of sample
        Value: Data
    """
    if not os.path.isfile(chrom_size_file):
        log.error(f"Chrom size file: {chrom_size_file} is not a valid file")
        return

    if not os.path.isdir(loop_data_dir):
        log.error(f"Loop dir: {loop_data_dir} is not a valid directory")
        return

    if not bedgraph_data_dir and not bigwig_data_dir:
        log.error('Need a directory with either bedgraph or bigwig files')
        return

    if bedgraph_data_dir and not os.path.isdir(bedgraph_data_dir):
        log.error(f"bedgraph dir: {bedgraph_data_dir} is not a valid directory")
        return

    if bigwig_data_dir and not os.path.isdir(bigwig_data_dir):
        log.error(f"bigwig dir: {bigwig_data_dir} is not a valid directory")
        return

    loop_data_dict = OrderedDict()

    log.info(os.listdir(loop_data_dir))
    for loop_file_name in os.listdir(loop_data_dir):
        if loop_file_name.endswith('.BE3') or loop_file_name.endswith('.cis'):
            sample_name = loop_file_name.split('.')[0]
            log.info(f'Loading {sample_name} ...')

            loop_file_path = os.path.join(loop_data_dir, loop_file_name)

            bedgraph = None
            if bedgraph_data_dir:
                for bedgraph_file_name in os.listdir(bedgraph_data_dir):
                    # Extra period from confusion with pooled data sets?
                    if f'{sample_name}.' in bedgraph_file_name and \
                            bedgraph_file_name.lower().endswith('bedgraph'):
                        bedgraph_file_path = os.path.join(bedgraph_data_dir,
                                                          bedgraph_file_name)

                        bedgraph = BedGraph(chrom_size_file, bedgraph_file_path,
                                            chroms_to_load=chroms_to_load,
                                            ignore_missing_bp=False,
                                            min_value=min_bedgraph_value)
                        break
            elif bigwig_data_dir:
                for bigwig_file_name in os.listdir(bigwig_data_dir):
                    # Extra period from confusion with pooled data sets?
                    if f'{sample_name}.' in bigwig_file_name and \
                            bigwig_file_name.lower().endswith('bigwig'):
                        bigwig_file_path = os.path.join(bigwig_data_dir,
                                                        bigwig_file_name)

                        bedgraph = BedGraph(chrom_size_file, bigwig_file_path,
                                            chroms_to_load=chroms_to_load,
                                            ignore_missing_bp=False,
                                            min_value=min_bedgraph_value)
                        break

            if bedgraph is None:
                if bigwig_data_dir:
                    log.error(f"{sample_name}'s bigwig file is not in "
                              f"{bigwig_data_dir}. Skipping")
                elif bedgraph_data_dir:
                    log.error(f"{sample_name}'s bedgraph file is not in "
                              f"{bedgraph_data_dir}. Skipping")
                continue

            gld = GenomeLoopData(chrom_size_file, loop_file_path, bedgraph,
                                 min_loop_value=min_loop_value,
                                 chroms_to_load=chroms_to_load)
            loop_data_dict[gld.sample_name] = gld

    return loop_data_dict


def preprocess(loop_dict, window_size=None):
    for sample_name in loop_dict:
        loop_dict[sample_name].preprocess(window_size)
