from collections import OrderedDict
from prettytable import PrettyTable
import numpy as np
import os
import logging
import math
import matplotlib.pyplot as plt

from .ChromLoopData import ChromLoopData, PEAK_MAX_VALUE_INDEX, MIN_PEAK_VALUE

VERSION = 17
log = logging.getLogger()
log_bin = logging.getLogger('bin')

# Missing in many miseq peak files
CHROMS_TO_IGNORE = ['chrY', 'chrM']

DEFAULT_NUM_PEAKS = 60


class GenomeLoopData:
    """
    A class used to represent a sample

    Attributes
    ----------
    species_name : str
        Name of the sample's species (hg38, mm10, ...)
    sample_name : str
        Name of the sample (LHH0061, LHH0061_0061H, ...)
    peak_dict : dict(str, list)
        Key: Name of chromosome (chr1, chr2, ...)
        Value: List of peaks in chromosome
        Peak format: [start, end, length, max_value, mean_value (optional)]
    chrom_dict : dict(str, ChromLoopData)
        Key: Name of chromosome
        Value: ChromLoopData object
    """

    def __init__(self, chrom_size_file_path, loop_file_path, bedgraph,
                 peak_dict, chroms_to_load=None, min_loop_value=0):
        """
        Initializes all chromosomes and adds loops to them from given file.

        Finds peak max from bedgraph

        Parameters
        ----------
        chrom_size_file_path : str
            File containing the base pair size of each chromosome to use
        loop_file_path : str
            File containing loops in format:
            chrom1  start1   end1 chrom2  start2   end2 pet_count
        bedgraph : BedGraph
            The bedgraph file for this sample (from pyBedGraph)
        peak_dict : dict(str, list)
            Key: Name of chromosome (chr1, chr2, ...)
            Value: List of peaks in chromosome
            Peak format: [start, end, length]
        chroms_to_load : list, optional
             List of names of chromosome to load (default is None)
        min_loop_value : int, optional
            Minimum loop value (PET count) to include (default is 0)
        """

        # Prints peak_dict which is too large to be meaningful
        # log.debug(locals())

        self.species_name = os.path.basename(chrom_size_file_path).split('.')[0]
        self.sample_name = os.path.basename(loop_file_path).split('.')[0]

        self.total_samples = 0

        self.peak_dict = {}

        # Find values for each peak since peak caller was not accurate
        for chrom_name, peak_chrom in peak_dict.items():
            if not bedgraph.has_chrom(chrom_name):
                log.warning(f'{bedgraph.name} does not have {chrom_name}')
                continue

            bedgraph.load_chrom_data(chrom_name)
            start_list = [x[0] for x in peak_chrom]
            end_list = [x[1] for x in peak_chrom]
            max_list = \
                bedgraph.stats(start_list=start_list, end_list=end_list,
                               chrom_name=chrom_name, stat='max')
            mean_list = \
                bedgraph.stats(start_list=start_list, end_list=end_list,
                               chrom_name=chrom_name, stat='mean')
            for i in range(max_list.size):
                peak_chrom[i].append(max_list[i])
                peak_chrom[i].append(mean_list[i])
            bedgraph.free_chrom_data(chrom_name)

            self.peak_dict[chrom_name] = peak_dict[chrom_name]

        # Initialize all chromosomes to be loaded
        self.chrom_dict = {}
        with open(chrom_size_file_path) as in_file:
            for line in in_file:
                line = line.strip().split()
                chrom_name = line[0]
                if chroms_to_load and chrom_name not in chroms_to_load:
                    continue

                if chrom_name in CHROMS_TO_IGNORE:
                    continue

                if chrom_name not in peak_dict:
                    continue

                chrom_size = int(line[1])

                self.chrom_dict[chrom_name] = \
                    ChromLoopData(chrom_name, chrom_size, self.sample_name)

        with open(loop_file_path) as in_file:
            loop_anchor_list = []
            for line in in_file:
                line = line.strip().split()
                chrom_name = line[0]
                if chrom_name not in self.chrom_dict:
                    continue

                loop_value = int(line[6])
                if loop_value < min_loop_value:
                    continue

                # head interval
                loop_start1 = int(line[1])
                loop_end1 = int(line[2])

                # tail anchor
                loop_start2 = int(line[4])
                loop_end2 = int(line[5])

                self.chrom_dict[chrom_name].add_loop(loop_start1, loop_end1,
                                                     loop_start2, loop_end2,
                                                     loop_value)

                head_interval = loop_end1 - loop_start1
                tail_interval = loop_end2 - loop_start2

                loop_anchor_list.append(head_interval)
                loop_anchor_list.append(tail_interval)

            log.debug(f'Anchor mean width: {np.mean(loop_anchor_list)}')

        # Get rid of chroms that had problems initializing
        to_remove = []
        for chrom_name in self.chrom_dict:
            if self.chrom_dict[chrom_name].finish_init(bedgraph):
                self.total_samples += \
                    np.sum(self.chrom_dict[chrom_name].value_list)
            else:
                to_remove.append(chrom_name)

        # Chromosomes with no loops or other random problems
        for chrom_name in to_remove:
            del self.chrom_dict[chrom_name]

    def preprocess(self, num_peaks=DEFAULT_NUM_PEAKS, both_peak_support=False,
                   kept_dir=None, base_chrom='chr1'):
        """
        Preprocess all the chromosomes in this object.

        Removes all problematic chromosomes (not enough loops, etc...). Keeps
        only num_peaks peaks in each list in peak_dict

        Parameters
        ----------
        num_peaks : int, optional
            The number of peaks to use when filtering. Only to be used with chr1
            since other chromosomes will be dependent on min peak used from chr1
            (default is DEFAULT_NUM_PEAKS)
        both_peak_support : bool, optional
            Whether to only keep loops that have peak support on both sides
            (default is False)
        kept_dir : str, optional
            Directory to output kept peaks and filters
            (default is None)

        Returns
        ------
        int
            Number of kept loops in all chromosomes
        """

        for peak_list in self.peak_dict.values():
            peak_list.sort(key=lambda x: x[PEAK_MAX_VALUE_INDEX], reverse=True)

        if num_peaks < 1:
            log.error(f'num_peaks is not positive')
            return 0

        min_peak_value = 0
        # peak_num_ratio = 1
        # chr1_size = 1

        if base_chrom in self.chrom_dict and base_chrom in self.peak_dict and \
                num_peaks > 0:
            base_peak_list = self.peak_dict[base_chrom]

            if num_peaks > len(base_peak_list):
                num_peaks = len(base_peak_list)

            while base_peak_list[num_peaks - 1][PEAK_MAX_VALUE_INDEX] < MIN_PEAK_VALUE \
                    and num_peaks - 1 > 0:
                num_peaks -= 1

            log.debug(f'Number of peaks: {num_peaks}')

            min_peak_value = \
                self.peak_dict[base_chrom][num_peaks - 1][PEAK_MAX_VALUE_INDEX]
            log.debug(f'Min peak value from {base_chrom}: {min_peak_value}')
            # peak_num_ratio = num_peaks / len(self.peak_dict['chr1'])
            # chr1_size = self.chrom_dict['chr1'].size
        else:
            log.warning(f'Unable to set min_peak_value since {base_chrom} '
                        f'is not available or num_peaks is not positive')

        to_remove = []
        for name, chrom_data in self.chrom_dict.items():

            # Keep only peaks in each chromosome above min_peak_value
            peak_list = self.peak_dict[name]
            if min_peak_value > 0:
                for i in range(len(peak_list)):
                    if peak_list[i][PEAK_MAX_VALUE_INDEX] < min_peak_value:
                        log.debug(f'Filtered out {len(peak_list) - i} peaks')
                        self.peak_dict[name] = self.peak_dict[name][:i]
                        break
            # peak_num_ratio = chrom_data.size / chr1_size
            # num_peaks_to_keep = math.ceil(len(peak_list) * peak_num_ratio)
            # num_peaks_to_keep = math.ceil(num_peaks * peak_num_ratio)
            # self.peak_dict[name] = self.peak_dict[name][:num_peaks_to_keep]

            if len(self.peak_dict[name]) == 0:
                log.warning(f'{name} has no peaks above {min_peak_value}')
                # log.warning(f'{name} total peaks: {len(peak_list)}, '
                #             f'ratio: {peak_num_ratio}')
                to_remove.append(name)
                continue

            if not chrom_data.preprocess(self.peak_dict[name],
                                         both_peak_support=both_peak_support):
                to_remove.append(name)

        # Remove problematic chromosomes
        for name in to_remove:
            del self.chrom_dict[name]

        if kept_dir and not os.path.isfile(
                f'{kept_dir}/{self.sample_name}.{num_peaks}.peaks'):
            if not os.path.isdir(kept_dir):
                os.mkdir(kept_dir)

            with open(f'{kept_dir}/{self.sample_name}.{num_peaks}.peaks', 'w+') \
                    as out_file:
                for name, peak_list in self.peak_dict.items():
                    for peak in peak_list:
                        out_file.write(f'{name}\t{peak[0]}\t{peak[1]}\t'
                                       f'{peak[PEAK_MAX_VALUE_INDEX]}\n')

            with open(f'{kept_dir}/{self.sample_name}.{num_peaks}.loops', 'w+') \
                    as out_file:
                for name, chrom_data in self.chrom_dict.items():
                    for i in chrom_data.kept_indexes:
                        out_file.write(
                            f'{chrom_data.name}\t'
                            f'{chrom_data.start_anchor_list[0][i]}\t'
                            f'{chrom_data.start_anchor_list[1][i]}\t'
                            f'{chrom_data.name}\t'
                            f'{chrom_data.end_anchor_list[0][i]}\t'
                            f'{chrom_data.end_anchor_list[1][i]}\t'
                            f'{chrom_data.pet_count_list[i]}\t'
                            f'{chrom_data.value_list[i]}\n')

        total_kept_loops = 0
        for name, chrom_loop in self.chrom_dict.items():
            total_kept_loops += chrom_loop.filtered_numb_values
        return total_kept_loops

    def compare(self, o_loop_data, bin_size, window_size, window_index=None,
                chroms_to_compare=None, is_rep=False):
        """
        Compares this sample against another sample

        Gets the values for each window in each chromosome and combines that
        into a genome-wide value.

        Parameters
        ----------
        o_loop_data : GenomeLoopData
            The sample to compare against
        bin_size : int
            Determines which loops are the same by putting them into bins
        window_size : int
            Cuts off all loops that are not entirely inside the window
        window_index : int, optional
            Determines a specific window index to look at (Default is None)
        chroms_to_compare : list(str), optional
            A list of chromosomes to compare (Default is All)
        is_rep : bool, optional
            Debugging purposes

        Returns
        ------
        dict
            graph_type : str
                Information about which comparison method was used
                Mainly useless since no comparisons to account for different
                sequencing depth are used
            rep : str
                Combined unweighted value of all chromosomes
            w_rep : str
                Combined weighted value of all chromosomes
            main : str
                The reproducibility value of this comparison (w_rep)
        """

        # my_kept_loops = self.preprocess(peak_percent_kept)
        # o_kept_loops = o_loop_data.preprocess(peak_percent_kept)
        #
        # my_loops = (self, my_kept_loops)
        # o_loops = (o_loop_data, o_kept_loops)
        # lesser_numb = min(my_loops, o_loops, key=lambda x: x[1])
        # greater_numb = max(my_loops, o_loops, key=lambda x: x[1])
        # if lesser_numb[1] / greater_numb[1] < 0.75:
        #     lesser_numb[0].preprocess(peak_percent_kept, greater_numb[1])

        # Default: Compare all the chromosomes
        if chroms_to_compare is None:
            chroms_to_compare = list(self.chrom_dict.keys())

        chrom_value_table = \
            PrettyTable(['chrom', 'emd_value', 'j_value'])

        chrom_value_list = []
        emd_dist_list = []
        j_value_list = []
        log.info(f'Chroms to compare: {chroms_to_compare}')
        for chrom_name in chroms_to_compare:

            if chrom_name not in self.chrom_dict:
                log.warning(f'{chrom_name} is not in {self.sample_name}. '
                            f'Skipping {chrom_name}')
                continue

            if chrom_name not in o_loop_data.chrom_dict:
                log.warning(f'{chrom_name} is in {self.sample_name} but '
                            f'not in {o_loop_data.sample_name}. Skipping '
                            f'{chrom_name}')
                continue

            log.info(f"Comparing {chrom_name} ...")

            # Compare for all windows in chrom
            chrom_size = self.chrom_dict[chrom_name].size
            value_dict_list = []
            numb_windows = math.ceil(chrom_size / window_size)
            for k in range(numb_windows):

                # If there is a specified window, just compare that
                if window_index is not None:
                    k = window_index

                window_start = window_size * k
                window_end = window_size * (k + 1)
                if window_end > chrom_size:
                    window_end = chrom_size

                value_dict_list.append(
                    self.chrom_dict[chrom_name].compare(
                        o_loop_data.chrom_dict[chrom_name], window_start,
                        window_end, bin_size, window_size, is_rep=is_rep))

                if window_index is not None:
                    break

            emd_dist_list += [x['emd_dist'] for x in value_dict_list]
            emd_values = [x['emd_value'] for x in value_dict_list]
            j_values = [x['j_value'] for x in value_dict_list]
            j_value_list += j_values
            chrom_value = OrderedDict()
            # for value_dict in value_dict_list:
            #    assert chrom_value['graph_type'] == value_dict['graph_type']
            try:  # Weigh value from each bin according to max loop in graph
                chrom_value['emd_value'] = \
                    np.average(emd_values, weights=[x['w'] for x in
                                                    value_dict_list])
                chrom_value['j_value'] = \
                    np.average(j_values, weights=[x['w'] for x in
                                                  value_dict_list])
            except ZeroDivisionError:  # sum of weights == 0
                log.exception(f"No loops were found in either graphs. Skipping"
                              f"{chrom_name}")
                continue

            # chrom_value = self.chrom_dict[chrom_name].compare(
            #     o_loop_data.chrom_dict[chrom_name],
            #     self.peak_dict[chrom_name] + o_loop_data.peak_dict[chrom_name],
            #     is_rep=is_rep)

            chrom_value_list.append(chrom_value)

            log.debug(f'{chrom_name} values: {chrom_value}')

            chrom_value_table.add_row([
                chrom_name, round(chrom_value['emd_value'], 5),
                round(chrom_value['j_value'], 5)])
            log_bin.info(chrom_value_table)
            chrom_value_table.clear_rows()

        # Print density of emd_dist
        # plt.title(
        #     f'{self.sample_name} vs. {o_loop_data.sample_name} EMD dist density')
        # plt.xlabel('EMD Dist')
        # plt.ylabel('Density')
        # plt.hist(emd_dist_list, bins=int(1 + np.log2(len(emd_dist_list))),
        #          density=True)
        # plt.savefig(
        #     f'graphs/{self.sample_name}_{o_loop_data.sample_name}_emd_dist_density')
        # plt.close()
        #
        # plt.title(
        #     f'{self.sample_name} vs. {o_loop_data.sample_name} j_value density')
        # plt.xlabel('j_value')
        # plt.ylabel('Density')
        # plt.hist(j_value_list, bins=int(1 + np.log2(len(j_value_list))),
        #          density=True)
        # plt.savefig(
        #     f'graphs/{self.sample_name}_{o_loop_data.sample_name}_j_value_density')
        # plt.close()

        log.debug(chrom_value_list)

        # Weigh value from each chromosome equally
        avg_value = {
            'emd_value': np.mean([x['emd_value'] for x in chrom_value_list]),
            'j_value': np.mean([x['j_value'] for x in chrom_value_list])
        }
        log.debug(avg_value)

        return avg_value

    def find_diff_loops(self, o_loop_data, bin_size, window_size, window_index=None,
                        chroms_to_diff=None, start_index=None, end_index=None,
                        output_dir='diff_loops'):

        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        if chroms_to_diff is None:
            chroms_to_diff = list(self.chrom_dict.keys())
        log.info(f'Chroms to compare: {chroms_to_diff}')

        for chrom_name in chroms_to_diff:

            if chrom_name not in self.chrom_dict:
                log.warning(f'{chrom_name} is not in {self.sample_name}. '
                            f'Skipping {chrom_name}')
                continue

            if chrom_name not in o_loop_data.chrom_dict:
                log.warning(f'{chrom_name} is in {self.sample_name} but '
                            f'not in {o_loop_data.sample_name}. Skipping '
                            f'{chrom_name}')
                continue

            log.info(f"Finding different loops for {chrom_name} ...")

            # Compare for all windows in chrom
            chrom_size = self.chrom_dict[chrom_name].size

            if start_index is None:
                start_index = 0

            if end_index is None:
                end_index = chrom_size

            if end_index < start_index:
                log.error(f"Given interval: ({start_index}, {end_index}) to "
                          f"diff. Cannot compute when start is greater than end.")
                return

            numb_windows = math.ceil((end_index - start_index) / window_size)

            diff_areas = []
            graph_arr = ([], [], [], [])

            output_file_name = f'{self.sample_name}_{o_loop_data.sample_name}'
            for k in range(numb_windows):

                # If there is a specified window, just compare that
                if window_index is not None:
                    k = window_index

                window_start = window_size * k + start_index
                window_end = window_start + window_size
                if window_end > end_index:
                    window_end = end_index

                diff_areas += self.chrom_dict[chrom_name].find_diff_loops(
                    o_loop_data.chrom_dict[chrom_name], window_start,
                    window_end, bin_size, window_size,
                    self.peak_dict[chrom_name] + o_loop_data.peak_dict[chrom_name],
                    self.total_samples, o_loop_data.total_samples, graph_arr)

                if window_index is not None:
                    break

            log.info(f'Plotting scatter plot with {len(graph_arr[0])} points')
            plt.scatter(graph_arr[0], graph_arr[1])
            plt.title(f'{output_file_name}_chr1')
            plt.savefig(f'diff_loops/{output_file_name}_values')
            plt.close()

            diff_areas.sort(key=lambda l: l[2], reverse=True)
            with open(f'{output_dir}/{output_file_name}.loops', 'w') as out_file:
                for area in diff_areas:
                    out_file.write(f'{area[0]}\t{area[1]}\t{area[2]}\n')
