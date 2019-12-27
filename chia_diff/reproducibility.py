import os
from collections import OrderedDict
import numpy as np
import csv
from prettytable import PrettyTable
import logging
from pyBedGraph import BedGraph
from .GenomeLoopData import GenomeLoopData, DEFAULT_NUM_PEAKS

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


def output_to_csv(scores, output_file):
    with open(output_file, 'w+') as out_file:
        header = ['Sample Name'] + list(scores.keys())
        writer = csv.DictWriter(out_file, fieldnames=header)

        writer.writeheader()
        for sample_name, sample_scores in scores.items():
            writer.writerow(sample_scores)

    log.info(f"Results have been written to {output_file}")


def compare(sample_dict, known_replicates=None, specified_comparisons=None,
            **kwargs):
    """
    Compares all samples in the dictionary against each other.

    Parameters
    ----------
    sample_dict : dict(str, GenomeLoopData)
        Key: Name of sample
        Value: Sample data
    known_replicates: list, optional
        2D List of known replicates in format:
        [[sample1, sample2], [sample1, sample3], ...]
        Default is None
    specified_comparisons : list, optional
        2D List of comparisons to make in format:
        [[sample1, sample2], [sample1, sample3], ...]
        Default is None
    kwargs :
        Extra arguments for compare function in GenomeLoopData

    Returns
    -------
    dict
        Contains reproducibility values for known replicates
    dict
        Contains reproducibility values for unknown samples
    OrderedDict
        Contains emd_values for every comparison made
    OrderedDict
        Contains j_values for every comparison made
    """
    log.info(f"Comparing {[x for x in sample_dict]}\nArguments: {kwargs}")

    non_rep_values = {}
    rep_values = {}

    if known_replicates is None:
        known_replicates = REPLICATES

    len_dict = len(sample_dict)
    keys = list(sample_dict.keys())
    emd_scores = OrderedDict()  # To easily output in .csv format
    j_scores = OrderedDict()  # To easily output in .csv format
    for key in keys:
        emd_scores[key] = OrderedDict()
        emd_scores[key][key] = 1
        emd_scores[key]['Sample Name'] = key

        j_scores[key] = OrderedDict()
        j_scores[key][key] = 1
        j_scores[key]['Sample Name'] = key

    for i in range(len_dict):
        for j in range(i + 1, len_dict):
            key1 = keys[i]
            key2 = keys[j]
            combined_keys = key1 + '_' + key2
            data1 = sample_dict[key1]
            data2 = sample_dict[key2]

            # Avoid comparing hg38 vs. mm10
            if data2.species_name != data1.species_name:
                continue

            if specified_comparisons:
                want_to_check = False
                for to_check in specified_comparisons:
                    if key1 in to_check and key2 in to_check:
                        want_to_check = True
                        break
                if not want_to_check:
                    continue

            is_known_rep = False
            for rep in known_replicates:
                if key1 in rep and key2 in rep:
                    is_known_rep = True
                    break

            log.info(f'{key1} vs. {key2}:')
            rep_dict = data1.compare(data2, **kwargs, is_rep=is_known_rep)

            # rep_dict = compare_func(data1, data2, **kwargs, is_rep=is_rep)

            emd_scores[key1][key2] = rep_dict['emd_value']
            emd_scores[key2][key1] = rep_dict['emd_value']
            emd_scores[key1][key2] = rep_dict['j_value']
            emd_scores[key2][key1] = rep_dict['j_value']

            # Separate reproducibility values into replicates and non-replicates
            if is_known_rep:
                rep_values[combined_keys] = rep_dict
            else:
                non_rep_values[combined_keys] = rep_dict

            log.info(f'{combined_keys} EMD: {rep_dict["emd_value"]}')
            log.info(f'{combined_keys} j_value: {rep_dict["j_value"]}')

    log.info(f"Compared {[x for x in sample_dict]}\nArguments: {kwargs}")

    return rep_values, non_rep_values, emd_scores, j_scores


def output_results(rep, non_rep, out_file_dir=None, desc_str=None):
    """
    Outputs results in readable text file table format

    Parameters
    ----------
    rep : dict(str, dict)
        Key: Combined comparison name (LHH0048_LHH0054L)
        Value: dict containing keys: 'emd_value' and/or 'j_value'
        Dictionary containing information for replicate comparisons
    non_rep : dict(str, dict)
        Key: Combined comparison name (LHH0048_LHH0054L)
        Value: dict containing keys: 'emd_value' and/or 'j_value'
        Dictionary containing information for non-replicate comparisons
    out_file_dir : str, optional
        Default is None
    desc_str : str, optional
        Used as file name to describe settings/parameters for this comparison
        Not optional if out_file_dir is not None
        Default is None
    """

    log.info(f"Number of known replicates: {len(rep)}")
    log.info(f"Number of non-replicates or unknown: {len(non_rep)}")

    for value_type in ['emd_value', 'j_value']:
        out_file = None
        out_file_path = None
        if out_file_dir:
            if not os.path.isdir(out_file_dir):
                os.mkdir(out_file_dir)

            if desc_str is None:
                out_file_path = os.path.join(out_file_dir, f'{value_type}.txt')
            else:
                out_file_path = os.path.join(out_file_dir,
                                             f'{desc_str}.{value_type}.txt')

            out_file = open(out_file_path, 'w')

        rep_table = PrettyTable(['Comparison', 'emd_value', 'j_value'])
        rep_table.sortby = value_type
        rep_table.reversesort = True

        for comparison_value in [rep, non_rep]:
            if len(comparison_value) == 0:
                continue

            for k, value_dict in comparison_value.items():
                rep_table.add_row([k, round(value_dict['emd_value'], 5),
                                   round(value_dict['j_value'], 5)])

            if comparison_value == rep:
                temp_str = f'Replicates sorted by {value_type}'
            else:
                temp_str = f'Non-Replicates sorted by {value_type}'
            temp_str += f'\n{rep_table.get_string()}'
            if out_file:
                out_file.write(temp_str + '\n')
            log.info(temp_str)
            rep_table.clear_rows()

        replicate_values = [x[value_type] for x in rep.values()]
        non_replicate_values = [x[value_type] for x in non_rep.values()]

        # No more statistics can be made without knowing replicates
        if len(non_replicate_values) == 0 or len(replicate_values) == 0:
            return

        min_diff = np.min(replicate_values) - np.max(non_replicate_values)
        avg_diff = np.mean(replicate_values) - np.mean(non_replicate_values)
        min_rep = np.min(replicate_values)
        max_non_rep = np.max(non_replicate_values)
        temp_str = f"Min replicate value: " \
                   f"{min(rep, key=lambda x: rep[x][value_type])} -> {min_rep}\n" \
                   f"Max non-replicate value: " \
                   f"{max(non_rep, key=lambda x: non_rep[x][value_type])} -> {max_non_rep}\n" \
                   f"Min diff between replicates and non-replicates: {min_diff}\n" \
                   f"Diff between replicate and non-replicate average: {avg_diff}"
        log.info(temp_str)

        if out_file_path:
            out_file.write(temp_str + '\n')
            log.info(f"Results have been written to {out_file_path}")
            out_file.close()


def read_data(loop_data_dir, chrom_size_file, peak_data_dir,
              bigwig_data_dir=None, bedgraph_data_dir=None,
              min_loop_value=0, min_bedgraph_value=0, chroms_to_load=None):
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
    peak_data_dir
        Directory with peak files. File names must include "narrowpeak" or
        "broadpeak" case insensitive.
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

    if not os.path.isdir(peak_data_dir):
        log.error(f"Peak dir: {peak_data_dir} is not a valid directory")
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

            peak_dict = None
            for peak_file_name in os.listdir(peak_data_dir):
                if not peak_file_name.lower().endswith('peak') or \
                        sample_name not in peak_file_name:
                    continue

                # Sort of useless now since we don't use value from peak caller
                is_narrowPeak = False
                if 'narrowpeak' in peak_file_name.lower():
                    is_narrowPeak = True
                elif 'broadpeak' in peak_file_name.lower():
                    is_narrowPeak = False
                else:
                    log.error(f"{peak_file_name} is an unknown peak file")

                peak_file_path = os.path.join(peak_data_dir, peak_file_name)
                peak_dict = read_peak_file(peak_file_path, is_narrowPeak)
                break

            if peak_dict is None:
                log.error(f"{sample_name}'s peak file is not in "
                          f"{peak_data_dir}. Skipping")
                continue

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
                                 peak_dict, min_loop_value=min_loop_value,
                                 chroms_to_load=chroms_to_load)
            loop_data_dict[gld.sample_name] = gld

    return loop_data_dict


def preprocess(loop_dict, num_peaks=DEFAULT_NUM_PEAKS, both_peak_support=False,
               kept_dir=None, base_chrom='chr1'):
    for sample_name in loop_dict:
        loop_dict[sample_name].preprocess(num_peaks,
                                          both_peak_support=both_peak_support,
                                          kept_dir=kept_dir,
                                          base_chrom=base_chrom)


# Deprecated
def read_bedgraphs(data_directory, chrom_size_file, min_value=-1,
                   chrom_to_read=None):
    """
    Unused due to not having enough memory. Read one bedgraph at a time instead.
    """
    bedgraph_dict = {}

    for filename in os.listdir(data_directory):
        if filename.lower().endswith('.bedgraph') or filename.endswith('peaks'):
            file_path = os.path.join(data_directory, filename)
            bedgraph = BedGraph(chrom_size_file, file_path, chrom_to_read,
                                ignore_missing_bp=False, min_value=min_value)
            bedgraph_dict[bedgraph.name] = bedgraph

    return bedgraph_dict


def read_peak_file(peak_file_path, is_narrowPeak=False):
    """
    Finds the start and ends of every peak in chromosome for one sample.

    File format must have at least 3 columns with the first 3 being:
    chrom   start   end

    Parameters
    ----------
    peak_file_path : str
        File path of peak file
    is_narrowPeak : bool, optional
        Useless for now since only the chrom/start/end is taken, not the value

    Returns
    -------
    dict(str, 2D list)
        Key: chrom name
        Value: list of [start, end, end - start]
    """
    peak_dict = {}

    if is_narrowPeak:
        value_index = 8
    else:  # .broadPeak files
        value_index = 6

    with open(peak_file_path) as peak_file:
        for line in peak_file:
            data = line.split()
            chrom_name = data[0]

            try:
                peak_start = int(data[1])
                peak_end = int(data[2])
            except ValueError:  # Sometimes have peaks with 1+E08 as a value
                log.error(f'Invalid peak: {line}')
                continue

            if chrom_name not in peak_dict:
                peak_dict[chrom_name] = []

            peak_dict[chrom_name].append([peak_start, peak_end,
                                          peak_end - peak_start])
    return peak_dict
