import os
from collections import OrderedDict
import logging
from pyBedGraph import BedGraph
import itertools
import prettytable
import csv

log = logging.getLogger()

VERSION = 12

COMPARISON_BASE_WEIGHT = 100

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

    from .GenomeLoopData import GenomeLoopData
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


def combine_random_walks(path_popularity, window_size):
    for sample_name, sample in path_popularity.items():

        for chrom_name in sample:
            chrom_popularity = path_popularity[sample_name][chrom_name]
            comb_chrom_popularity = {}

            window_starts = sorted(list(chrom_popularity.keys()))

            for i, window_start in enumerate(window_starts):

                window_popularity = chrom_popularity[window_start]

                for pair in window_popularity:
                    comb_chrom_popularity[pair] = window_popularity[pair]

                    # Double count the beginning window (and end?)
                    if window_start == 0:
                        assert i == 0

                        anchors = pair.split('-')
                        start = int(anchors[0])
                        end = int(anchors[1])

                        if start < window_size / 2 or end < window_size / 2:
                            comb_chrom_popularity[pair] += \
                                window_popularity[pair]

                    if i == len(window_starts) - 1:
                        anchors = pair.split('-')
                        start = int(anchors[0])
                        end = int(anchors[1])

                        if start > window_start + window_size / 2 or end > \
                            window_start + window_size / 2:
                            comb_chrom_popularity[pair] += \
                                window_popularity[pair]

            path_popularity[sample_name][chrom_name] = comb_chrom_popularity


def output_path_start_counts(chrom, window_start, window_end, loop_bin_size,
                             walk_iter, start_nodes, start_type):

    if not os.path.isdir('random_walks/path_start_count'):
        os.mkdir('random_walks/path_start_count')

    with open(f'random_walks/path_start_count/{chrom.sample_name}.{chrom.name}:'
              f'{window_start}-{window_end}.walk_iter={walk_iter}.'
              f'path_start_count.{start_type}.txt',
              'w') as out_file:
        start_popularity = {}
        for node in start_nodes:
            if node not in start_popularity:
                start_popularity[node] = 0
            start_popularity[node] += 1

        start_popularity = \
            {k: v for k, v in sorted(start_popularity.items(),
                                     key=lambda l: l[1], reverse=True)}

        for node in start_popularity:
            start_index = node * loop_bin_size + window_start
            out_file.write(
                f'{chrom.name}\t{start_index}\t{start_index + loop_bin_size}\t'
                f'{start_popularity[node]}\n')


def output_node_weights(chrom, window_start, window_end, used_loop_bins,
                        walk_iter, node_weights):
    if not os.path.isdir('random_walks/node_weights'):
        os.mkdir('random_walks/node_weights')
    with open(f'random_walks/node_weights/{chrom.sample_name}.{chrom.name}:'
              f'{window_start}-{window_end}.walk_iter={walk_iter}.'
              f'node_weights.txt', 'w') as out_file:
        for i in range(len(used_loop_bins)):
            out_file.write(f'{used_loop_bins[i]}\t{node_weights[i]}\n')


def output_random_walk_loops(window_size, loop_bin_size, walk_iter,
                             path_popularity, walk_type):

    if not os.path.isdir('random_walks/bin_loops'):
        os.mkdir('random_walks/bin_loops')

    for sample_name, sample_paths in path_popularity.items():
        with open(f'random_walks/bin_loops/{sample_name}.'
                  f'window_size={window_size}.loop_bin_size={loop_bin_size}.'
                  f'walk_iter={walk_iter}.{walk_type}.forward.loops',
                  'w') as forward_file, \
                open(f'random_walks/bin_loops/{sample_name}.'
                     f'window_size={window_size}.loop_bin_size={loop_bin_size}.'
                     f'walk_iter={walk_iter}.{walk_type}.back.loops',
                     'w') as back_file:

            for chrom_name, chrom_paths in sample_paths.items():

                for key in chrom_paths:
                    anchors = key.split('-')
                    start = int(anchors[0])
                    end = int(anchors[1])

                    if end > start:
                        forward_file.write(
                            f'{chrom_name}\t{start}\t{start + loop_bin_size}\t'
                            f'{chrom_name}\t{end}\t{end + loop_bin_size}\t'
                            f'{chrom_paths[key]}\n')

                    elif start > end:
                        back_file.write(
                            f'{chrom_name}\t{end}\t{end + loop_bin_size}\t'
                            f'{chrom_name}\t{start}\t{start + loop_bin_size}\t'
                            f'{chrom_paths[key]}\n')

                    else:
                        log.warning(f'Detected self loop: {key}. Maybe bug in '
                                    f'code. Should have removed when creating '
                                    f'adj list')


def output_random_walk_path(sample_name, chrom_name, window_start, window_end,
                            loop_bin_size, walk_iter, paths):
    if not os.path.isdir('random_walks/paths'):
        os.mkdir('random_walks/paths')

    with open(f'random_walks/paths/{sample_name}.{chrom_name}:'
              f'{window_start}-{window_end}.loop_bin_size={loop_bin_size}.'
              f'walk_iter={walk_iter}.txt', 'w') as out_file:
        for walk in paths:
            walk = '\t'.join([str(w) for w in walk])
            out_file.write(f'{walk}\n')


def output_coverage_bins(sample_name, chrom_name, coverage_bins,
                         coverage_bin_size):
    if not os.path.isdir('coverage_bins'):
        os.mkdir('coverage_bins')

    with open(f'coverage_bins/{sample_name}.{chrom_name}.'
              f'coverage_bins={coverage_bin_size}.txt', 'w') as out_file:
        for i in range(len(coverage_bins)):
            out_file.write(f'{i * coverage_bin_size}\t'
                           f'{(i + 1) * coverage_bin_size}\t'
                           f'{coverage_bins[i]}\n')


def compare_random_walks(path_popularity, walk_iter):
    """
    Makes a dict with key: pair_number, value: list of occurrences for each
    sample.
    """

    sample_list = list(path_popularity.keys())
    comparison_list = list(itertools.combinations(sample_list, 2))
    sample_list_order = []  # Keep order of samples

    popularity_dict = {}
    for sample_num, sample_name in enumerate(path_popularity):
        for chrom_name in path_popularity[sample_name]:

            if chrom_name not in popularity_dict:
                popularity_dict[chrom_name] = {}

            sample_list_order.append(sample_name)
            chrom_popularity = path_popularity[sample_name][chrom_name]
            combined_chrom_popularity = popularity_dict[chrom_name]

            for pair in chrom_popularity:
                if pair not in combined_chrom_popularity:

                    # Check if there were samples that were looked at before
                    combined_chrom_popularity[pair] = \
                            [0 for _ in range(sample_num)] + \
                            [chrom_popularity[pair]]
                else:
                    combined_chrom_popularity[pair].append(
                        chrom_popularity[pair])

            # Add 0's to pairs not in this sample
            for pair in combined_chrom_popularity:
                if len(combined_chrom_popularity[pair]) <= sample_num:
                    combined_chrom_popularity[pair].append(0)

    for chrom_name, chrom_popularity in popularity_dict.items():
        with open(f'random_walks/total_popularity_comparison.{chrom_name}.'
                  f'walk_iter={walk_iter}.pretty.txt', 'w') as pretty_file, \
            open(f'random_walks/total_popularity_comparison.{chrom_name}.'
                 f'walk_iter={walk_iter}.txt', 'w') as out_file:

            popularity_table = prettytable.PrettyTable()
            popularity_table.field_names = ['Pair'] + list(path_popularity) + [
                f'{x[0]}/{x[1]}' for x in comparison_list
            ]

            output_line = '\t'.join(popularity_table.field_names)
            out_file.write(f'{output_line}\n')

            for pair, popularity_list in chrom_popularity.items():
                for comparison in comparison_list:
                    sample1 = comparison[0]
                    sample2 = comparison[1]
                    sample1_index = sample_list_order.index(sample1)
                    sample2_index = sample_list_order.index(sample2)
                    sample1_popularity = popularity_list[sample1_index]
                    sample2_popularity = popularity_list[sample2_index]

                    # Weight higher popularities more than lower popularities
                    # when sorting
                    popularity_list.append(
                        (sample1_popularity + COMPARISON_BASE_WEIGHT) /
                        (sample2_popularity + COMPARISON_BASE_WEIGHT))
                output_line = '\t'.join(str(popularity_list))
                out_file.write(f'{output_line}\n')

                popularity_table.add_row([pair] + popularity_list)

            # TODO: Potentially customize the sort?
            popularity_table.sortby = popularity_table.field_names[
                len(path_popularity) + 1]
            popularity_table.reversesort = True

            pretty_file.write(popularity_table.get_string())
