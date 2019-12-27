import itertools
from math import ceil
import scipy.stats as sp
import time
from copy import deepcopy
import logging
from .chia_rep_util import *

from prettytable import PrettyTable

log = logging.getLogger()
log_bin = logging.getLogger('bin')

EMD_WEIGHT = 1
J_WEIGHT = 1
MIN_NUMB_LOOPS = 5
MAX_LOOP_LEN = 1000000  # 1mb

VERSION = 60

MAX_USHRT = 65535
MIN_RATIO_INCREASE = 1.1

PEAK_START_INDEX = 0
PEAK_END_INDEX = 1
PEAK_LEN_INDEX = 2
PEAK_MAX_VALUE_INDEX = 3
PEAK_MEAN_VALUE_INDEX = 4

MIN_PEAK_VALUE = 70

# The length of each normalization call
NORM_LEN = 100


def mypow(x, power):
    for i in range(power - 1):
        x *= x
    return x


def emd(p, q):
    """
    Finds the Earth Mover's Distance of two 1D arrays

    Parameters
    ----------
    p : 1D Numpy array, float64
    q : 1D Numpy array, float64

    Returns
    -------
    float/int
        The Earth Mover's Distance
    int
        The weight of this calculation
    """
    p_sum = p.sum()
    q_sum = q.sum()

    if p_sum == 0 and q_sum == 0:
        return 0, 0

    if p_sum == 0:
        return q.size, q_sum

    if q_sum == 0:
        return p.size, p_sum

    p = p / p_sum
    q = q / q_sum
    return c_emd(p, q, p.size), q_sum + p_sum


def jensen_shannon_divergence(p, q, base=2):
    """
    Finds the jensen-shannon divergence

    Parameters
    ----------
    p : 2D Numpy array, float64
        Graph of sample1
    q : 2D Numpy array, float64
        Graph of sample2
    base : int, optional
        Determines base to be used in calculating scipy.entropy (Default is 2)

    Returns
    -------
    float/int
        Jensen-Shannon divergence
    """
    assert p.size == q.size
    p_sum = p.sum()
    q_sum = q.sum()

    if p_sum == 0 and q_sum == 0:
        return 0

    if p_sum == 0:
        return 1

    if q_sum == 0:
        return 1

    p, q = p / p_sum, q / q_sum
    # log.info(np.info(p))
    # log.info(np.info(q))
    # log.info(p[p > 0.00000001])

    m = (p + q) / 2
    return sp.entropy(p, m, base=base) / 2 + sp.entropy(q, m, base=base) / 2


def get_matching_graphs(p, q):
    """
    Used with ChromLoopData.compare_only_match

    Modified array has values only where the other sample also has values. This
    was originally for miseq-hiseq comparisons where miseq could be missing
    important loops found in hiseq.

    Parameters
    ----------
    p : 2D Numpy array, float64
        Graph of sample1
    q : 2D Numpy array, float64
        Graph of sample2

    Returns
    -------
    list
        [[mod, orig], [mod, orig]]
    """
    assert p.size == q.size
    q1 = deepcopy(q)
    p1 = deepcopy(p)
    match_graphs(p, q, p1, q1, p[0].size)

    # Could probably be made simpler: return p1, q1
    return [[p1, p], [q1, q]]


def get_removed_area(chrom_size, chrom, o_chrom):
    """
    Gets removed area from both samples so no loops from that area are compared

    Parameters
    ----------
    chrom_size : int
        Size of chromosome
    chrom : ChromLoopData
    o_chrom : ChromLoopData

    Returns
    -------
    1D Numpy array, uint8
        Array marked with 1 in removed intervals
    """
    to_remove = np.array(
        [chrom.removed_intervals[0] + o_chrom.removed_intervals[0],
         chrom.removed_intervals[1] + o_chrom.removed_intervals[1]],
        dtype=np.int32)

    # Uses dtype unsigned char because bool doesn't work as well in Cython
    removed_area = np.zeros(chrom_size, dtype=np.uint8)
    for i in range(len(to_remove[0])):
        start = to_remove[0][i]
        end = to_remove[1][i]
        removed_area[start:end] = 1

    return removed_area


class ChromLoopData:
    """
    A class used to represent a chromosome in a sample

    Attributes
    ----------
    name : str
        Name of chromosome
    size : int
        Size of chromosome
    sample_name : str
        Name of the sample (LHH0061, LHH0061_0061H, ...)
    """

    def __init__(self, chrom_name, chrom_size, sample_name):
        """
        Parameters
        ----------
        chrom_name : str
            Name of chromosome
        chrom_size : int
            Size of chromosome
        sample_name : str
            Name of the sample (LHH0061, LHH0061_0061H, ...)
        """

        self.name = chrom_name
        self.size = chrom_size
        self.sample_name = sample_name

        self.start_anchor_list = [[], []]
        self.end_anchor_list = [[], []]
        self.start_list = None  # Later initialized with bedgraph
        self.end_list = None
        self.value_list = []
        self.pet_count_list = []
        self.numb_loops = 0
        self.removed_intervals = [[], []]  # Start/end anchors are on same peak
        self.start_list_peaks = None
        self.end_list_peaks = None

        self.norm_list = None

        self.filtered_start = []
        self.filtered_end = []
        self.filtered_values = []
        self.filtered_numb_values = 0
        self.filtered_anchors = []
        self.kept_indexes = []

        # Used in filter_with_peaks, keeps track of peaks for each loop
        self.peak_indexes = []
        self.peaks_used = None

        self.max_loop_value = 0

    def add_loop(self, loop_start1, loop_end1, loop_start2, loop_end2,
                 loop_value):

        self.start_anchor_list[0].append(loop_start1)
        self.start_anchor_list[1].append(loop_end1)
        self.end_anchor_list[0].append(loop_start2)
        self.end_anchor_list[1].append(loop_end2)

        self.value_list.append(loop_value)

        self.numb_loops += 1

    def finish_init(self, bedgraph):
        """
        Finishes the construction of this chromosome. Converts lists to numpy
        arrays and calls find_loop_anchor_points

        Parameters
        ----------
        bedgraph : BedGraph
            Used to find the anchor points of each loop

        Returns
        -------
        bool
            Whether the chromosome was successfully made
        """

        if self.numb_loops == 0:
            return False

        self.pet_count_list = np.asarray(self.value_list, dtype=np.uint32)
        self.value_list = np.asarray(self.value_list, dtype=np.float64)
        self.start_anchor_list = np.asarray(self.start_anchor_list,
                                            dtype=np.int32)
        self.end_anchor_list = np.asarray(self.end_anchor_list, dtype=np.int32)

        log.debug(f"Max PET count: {np.max(self.pet_count_list)}")

        # if not bedgraph.has_chrom(self.name):
        if self.name not in bedgraph.chromosome_map:
            log.warning(f"{self.name} was not found in corresponding bedgraph: "
                        f"{bedgraph.name}")
            return False

        self.find_loop_anchor_points(bedgraph)
        return True

    def find_loop_anchor_points(self, bedgraph):
        """
        Finds the exact loop anchor points.

        Finds peak values for each anchor and weighs the loop. Also finds loops
        that have overlapping start/end indexes due to close and long start/end
        anchors.

        Parameters
        ----------
        bedgraph : BedGraph
            Used to find the anchor points of each loop
        """

        log.info(f'Finding anchor points for {self.sample_name}\'s {self.name}'
                 f' from {bedgraph.name}')

        bedgraph.load_chrom_data(self.name)

        norm_start_test = np.arange(0, self.size, NORM_LEN, dtype=np.int32)
        norm_end_test = norm_start_test + NORM_LEN
        if norm_end_test[-1] > self.size:
            norm_end_test[-1] = self.size
        self.norm_list = bedgraph.stats(start_list=norm_start_test,
                                        end_list=norm_end_test,
                                        chrom_name=self.name, stat='max')
        self.norm_list += 1  # To avoid zeros

        # Get index of peaks in every anchor interval
        self.start_list = bedgraph.stats(start_list=self.start_anchor_list[0],
                                         end_list=self.start_anchor_list[1],
                                         chrom_name=self.name, stat='max_index')
        self.end_list = bedgraph.stats(start_list=self.end_anchor_list[0],
                                       end_list=self.end_anchor_list[1],
                                       chrom_name=self.name, stat='max_index')

        # Get peak value for every anchor interval
        start_list_peaks = bedgraph.stats(start_list=self.start_anchor_list[0],
                                          end_list=self.start_anchor_list[1],
                                          chrom_name=self.name, stat='max')
        end_list_peaks = bedgraph.stats(start_list=self.end_anchor_list[0],
                                        end_list=self.end_anchor_list[1],
                                        chrom_name=self.name, stat='max')
        self.start_list_peaks = start_list_peaks
        self.end_list_peaks = end_list_peaks
        bedgraph.free_chrom_data(self.name)

        start_list_peaks = start_list_peaks / start_list_peaks.sum()
        end_list_peaks = end_list_peaks / end_list_peaks.sum()

        # Merge peaks that are close together
        # for i in range(self.numb_values):
        #     for j in range(i, self.numb_values):
        #         pass

        for i in range(self.numb_loops):
            loop_start = self.start_list[i]
            loop_end = self.end_list[i]

            # Remove anchors that have the same* peak
            # Keep indexes of loop length to avoid comparisons in interval
            if not loop_start < loop_end:
                self.value_list[i] = 0

                # Removed interval goes from
                # (start of start anchor, end of end anchor)
                self.removed_intervals[0].append(self.start_anchor_list[0][i])
                self.removed_intervals[1].append(self.end_anchor_list[1][i])
                continue

            # Weigh each loop based on its corresponding bedgraph peak
            # peak_value = max(start_list_peaks[i], end_list_peaks[i])
            # peak_value = start_list_peaks[i] * end_list_peaks[i]
            # self.value_list[i] *= peak_value

            # Remove loops over a given threshold
            # loop_length = int(loop_end) - int(loop_start)
            # if loop_length > MAX_LOOP_LEN:
            #     self.value_list[i] = 0

        self.max_loop_value = np.max(self.value_list)

        # Should be very small due to peaks being weighted earlier
        log.debug(f"Max loop weighted value: {self.max_loop_value}")

    # May have more pre-processing to do besides filtering later?
    # Useless extra function otherwise
    def preprocess(self, peak_list, both_peak_support=False):
        return self.filter_with_peaks(peak_list, both_peak_support)

    def filter_with_peaks(self, peak_list, both_peak_support=False):
        """
        Filters out loops without peak support.

        Get coverage of peaks that have been chosen to be used. Find loops that
        are not within that coverage and filter them out.

        Parameters
        ----------
        peak_list : list(list)
            List of peaks to use
        both_peak_support : bool, optional
            Whether to only keep loops that have peak support on both sides
            (default is False)

        Returns
        -------
        bool
            Whether the chromosome had any problems when filtering
        """

        start_time = time.time()

        num_peaks = len(peak_list)
        min_peak_value = peak_list[-1][PEAK_MAX_VALUE_INDEX]

        log.info(f"Filtering {self.sample_name} {self.name} with "
                 f"{num_peaks} peaks...")
        log.debug(f"Top peaks: {peak_list[:3]}")
        log.debug(f"Bot peaks: {peak_list[-3:]}")
        log.debug(f'Min peak value: {min_peak_value}')

        # Get the coverage of each wanted peak
        # Could be used to find the specific peaks for every loop
        index_array = np.zeros(self.size, dtype=np.uint16)
        assert num_peaks < MAX_USHRT
        for i in range(num_peaks):
            peak_start = peak_list[i][0]
            peak_end = peak_list[i][1]
            index_array[peak_start:peak_end] = i + 1

        log.debug(f'Time: {time.time() - start_time}')

        numb_deleted = 0
        removed_loop_lengths = []
        removed_loop_values = []
        kept_loop_lengths = []
        self.kept_indexes = []
        self.peak_indexes = [[], []]
        self.filtered_start = []
        self.filtered_end = []
        self.filtered_values = []
        self.filtered_anchors = []

        # plt.title(f'{self.sample_name} Log10(loop span)')
        # plt.xlabel('log10(Loop Span)')
        # plt.ylabel('Density')
        # loop_spans = [self.end_list[i] - self.start_list[i] for i in range(self.numb_loops)]
        # plt.hist(loop_spans, bins=int(1 + np.log2(len(loop_spans))), density=True)
        # plt.savefig(f'{self.sample_name}_loop_span_all')
        # plt.close()

        for i in range(self.numb_loops):
            loop_start = self.start_list[i]
            loop_end = self.end_list[i]
            loop_value = self.value_list[i]

            # Loops that are too long can be considered noise
            if loop_end - loop_start > MAX_LOOP_LEN:
                continue

            # From overlapping anchors
            if loop_value == 0:
                continue

            if both_peak_support:
                to_keep = index_array[loop_start] and index_array[loop_end]
            else:
                to_keep = index_array[loop_start] or index_array[loop_end]

            if not to_keep:
                removed_loop_values.append(loop_value)
                numb_deleted += 1
                removed_loop_lengths.append(loop_end - loop_start)
                continue

            self.filtered_start.append(loop_start)
            self.filtered_end.append(loop_end)
            self.filtered_values.append(loop_value)
            self.filtered_anchors.append([self.start_anchor_list[0][i],
                                          self.start_anchor_list[1][i],
                                          self.start_list_peaks[i]])
            self.filtered_anchors.append([self.end_anchor_list[0][i],
                                          self.end_anchor_list[1][i],
                                          self.start_list_peaks[i]])
            self.kept_indexes.append(i)

            kept_loop_lengths.append(loop_end - loop_start)

            # Unused for now
            self.peak_indexes[0].append((
                peak_list[index_array[loop_start] - 1][0],
                peak_list[index_array[loop_start] - 1][1]))
            self.peak_indexes[1].append((
                peak_list[index_array[loop_end] - 1][0],
                peak_list[index_array[loop_end] - 1][1]))

        self.filtered_start = np.array(self.filtered_start, dtype=np.int32)
        self.filtered_end = np.array(self.filtered_end, dtype=np.int32)
        self.filtered_values = np.array(self.filtered_values)
        self.filtered_numb_values = self.filtered_start.size
        self.kept_indexes = np.array(self.kept_indexes, dtype=np.int32)

        log.debug(f'Total loops: {self.numb_loops}')
        log.debug(f"Number of loops removed: {numb_deleted}")
        log.info(f"Number of loops kept: {self.filtered_numb_values}")

        if self.filtered_numb_values == 0:
            log.warning(f"No loops left. Skipping")
            return False

        if numb_deleted > 0:
            log.debug(
                f'Avg loop length removed: {np.mean(removed_loop_lengths)}')
            log.debug(f'Avg loop value removed: {np.mean(removed_loop_values)}')
        else:
            log.debug(f'Avg loop length removed: N/A')
            log.debug(f'Avg loop value removed: N/A')
        log.debug(f'Avg loop length kept: {np.mean(kept_loop_lengths)}')
        log.debug(f'Avg loop value kept: {np.mean(self.filtered_values)}')
        log.debug(f'Largest loop value kept: {np.max(self.filtered_values)}')
        log.debug(f'Time taken: {time.time() - start_time}\n')

        return True

    def create_graph(self, loops, bin_size, window_size,
                     random=False, num_loops=0, to_debug=False):
        """
        Creates a bin-based graph to easily compare loops

        Parameters
        ----------
        loops : list
            List of loop indexes from self.filtered_*
        bin_size : int
        window_size : int
        random : bool, optional
            Randomly pick which loops to use (Default is False)
            Useless for now since different sequencing depth is ignored
        num_loops : int, optional
            Number of loops to use when making the graph
        to_debug : bool, optional
            Log loops used in the graph (Default is False)

        Returns
        -------
        Numpy 2D array
        """

        graph_len = ceil(window_size / bin_size)
        graph = np.zeros((graph_len, graph_len), dtype=np.float64)

        # So comparing a sample with no loops vs. a sample with loops doesn't
        # result in high reproducibility
        if num_loops == 0:
            num_loops = len(loops)
        indexes = None
        if random:
            indexes = np.random.choice(len(loops), num_loops, replace=False)

        num_loops_used = 0
        for i in range(num_loops):

            if indexes is not None:
                loop_index = loops[indexes[i]]
            else:
                loop_index = loops[i]

            value = self.filtered_values[loop_index]
            start = self.filtered_start[loop_index]
            end = self.filtered_end[loop_index]
            orig_start = start
            orig_end = end

            start = start % window_size
            end = end % window_size

            # loop_len[int((end - start) / self.bin_size)] += 1

            bin_start = int(start / bin_size)
            bin_end = int(end / bin_size)

            graph[bin_start][bin_end] += value

            # if bin_end < bin_start:
            #     log.error(
            #         f'{orig_start}\t{orig_end}\t{start}\t{end}\t{bin_start}\t{bin_end}')

            # Get the other side of the graph as well for emd calculation
            # graph[bin_end][bin_start] += value

            # Avoid double counting the middle
            # if bin_end != bin_start:
            #    graph[bin_end][bin_start] += value

            # Also get areas surrounding this loop
            # May not be needed with emd calculation
            # Helps with finding jensen-shannon
            for j in range(bin_start - 1, bin_start + 2):
                if j < 0 or j == graph_len:
                    continue
                for k in range(bin_end - 1, bin_end + 2):
                    if k < 0 or k == graph_len:
                        continue
                    graph[j][k] += value
                    # graph[k][j] += value
                    # if j != k:
                    #    graph[k][j] += value

            # if to_debug:
            #     log.debug(
            #         f'{self.sample_name}\t{orig_start}\t{orig_end}\t{value}')

            num_loops_used += 1

        # log.info(f"Number of loops in {self.sample_name} graph: {num_loops_used}")

        # plt.plot([x for x in range(len(loop_len))], [np.log(x) for x in loop_len])
        # plt.show()

        # log.info(f'Max value in graph: {np.max(graph)}')
        # return graph, total_PET_count / self.total_loop_value
        return graph

    def get_stats(self, graph, o_graph, o_chrom, graph_type):
        """
        Get the reproducibility stat.

        Jensen Shannon and EMD. Though EMD seems to be better.

        Parameters
        ----------
        graph : 2D Numpy array, float64
            Graph created from window from sample1
        o_graph : 2D Numpy array, float64
            Graph created from window from sample2
        o_chrom : ChromLoopData
            sample2
        graph_type : str
            Description of type of comparison

        Returns
        -------
        dict
            graph_type : str
                Information about which comparison method was used
            rep : str
                Reproducibility statistic for this window comparison
            w : str
                Weight of this window
        """

        max_emd_dist = graph[0].size - 1
        graph_size = graph[0].size
        log.debug(f'Graph size: {graph_size}')

        result = {'graph_type': graph_type}

        max_graph = np.max(graph)
        max_o_graph = np.max(o_graph)

        result['w'] = max_graph / self.max_loop_value + \
                      max_o_graph / o_chrom.max_loop_value

        if max_graph == 0 or max_o_graph == 0:
            if max_graph == 0:
                log.debug('No loops in sample A')
            else:
                log.debug('No loops in sample B')

            result['j_divergence'] = 1
            result['j_value'] = -1
            result['emd_value'] = -1
            result['linear_emd_value'] = -1
            result['emd_dist'] = 1
            return result

        log.debug('Loops in both samples')

        # with open(f'graphs/{self.sample_name}_{window_start}.csv', 'w') as out:
        #     sparse_mat = coo_matrix(graph)
        #     for row, col, value in zip(sparse_mat.row, sparse_mat.col, sparse_mat.data):
        #         out.write("({0}, {1})\t{2}\n".format(row, col, round(value, 6)))
        # with open(f'graphs/{o_chrom.sample_name}_{window_start}.csv', 'w') as out:
        #     sparse_mat = coo_matrix(graph)
        #     for row, col, value in zip(sparse_mat.row, sparse_mat.col, sparse_mat.data):
        #         out.write("({0}, {1})\t{2}\n".format(row, col, round(value, 6)))

        # total_weight = num_loops + num_o_loops
        # total_weight = graph.sum() + o_graph.sum()

        start_time = time.time()
        graph_flat = graph.flatten()
        o_graph_flat = o_graph.flatten()
        j_divergence = jensen_shannon_divergence(graph_flat, o_graph_flat)
        log.debug(f'Jensen-Shannon time: {time.time() - start_time}')

        # Make j_value range from -1 to 1
        j_value = 2 * (1 - j_divergence) - 1

        # complete_graph(graph, max_emd_dist)
        # complete_graph(o_graph, max_emd_dist)

        # Calculate emd for all rows and columns -> Take weighted average
        emd_distance_list = []
        emd_weight_list = []
        start_time = time.time()
        for k in range(graph[0].size):
            # row/col max_emd_dist from the 9-spread when creating graphs
            # Remember max_emd_dist = graph.size - 1

            # Finding number of blanks in row/column doesn't actually help so it
            # might just be simpler to take the emd of row/column - 9/12/19
            # numb_blanks = k - 2
            # if numb_blanks < 0:
            #     numb_blanks = 0
            # row_max_emd_dist = max_emd_dist - numb_blanks
            emd_dist, emd_weight = emd(graph[k], o_graph[k])
            # if emd_dist == graph_size:
            #     emd_dist = row_max_emd_dist
            # emd_distance_list.append(emd_dist / row_max_emd_dist)
            emd_distance_list.append(emd_dist)
            emd_weight_list.append(emd_weight)

            # Equivalent to max_emd_dist - (k + 2)
            # numb_blanks = graph_size - (k + 3)
            # if numb_blanks < 0:
            #     numb_blanks = 0
            # col_max_emd_dist = max_emd_dist - numb_blanks
            emd_dist, emd_weight = emd(graph[:, k], o_graph[:, k])
            # if emd_dist == graph_size:
            #     emd_dist = col_max_emd_dist
            # emd_distance_list.append(emd_dist / col_max_emd_dist)
            emd_distance_list.append(emd_dist)
            emd_weight_list.append(emd_weight)

        log.debug(f'EMD time: {time.time() - start_time}')
        max_emd_weight = np.max(emd_weight_list)

        if max_emd_weight == 0:
            overall_emd_dist = 0
        else:
            overall_emd_dist = np.average(emd_distance_list,
                                          weights=emd_weight_list)
        # emd_dist = np.mean(emd_distance_list)

        # Change from 0 to 1 range to -1 to 1
        # 1 is bad rep -> -1 is bad rep
        # 0 is good rep -> 1 is good rep
        # emd_value = 2 * (1 - overall_emd_dist) - 1
        # Since most values are in bottom half, don't scale linearly
        emd_value = 2 * mypow(overall_emd_dist - max_emd_dist, 2) / \
                    mypow(max_emd_dist, 2) - 1
        # emd_value = 2 * mypow(overall_emd_dist - 1, 2) - 1

        # Linear scale
        # emd_value = 1 - 2 / max_emd_dist * overall_emd_dist
        # linear_emd_value = 1 - 2 * overall_emd_dist

        if max_emd_weight == 0 and result['w'] != 0:
            log.error(f'Total Weight: {result["w"]} with 0 emd dist')

        # result['rep'] = emd_value * EMD_WEIGHT + j_value * J_WEIGHT
        result['j_divergence'] = j_divergence
        result['j_value'] = j_value
        result['emd_value'] = emd_value
        result['emd_dist'] = overall_emd_dist
        # result['linear_emd_value'] = linear_emd_value
        # table.add_row([graph_type, j_value, emd_value, result['w']])

        return result

    def compare_randomly_choose(self, loops, o_loops, o_chrom, bin_size,
                                window_size, inner_table, num_iterations=10):
        """
        Randomly choose which loops to compare to better match samples with
        differing sequencing depth

        (Deprecated) (Doesn't work?)
        """

        start_time = time.time()

        num_loops_to_choose = min(len(loops), len(o_loops))

        value_table = PrettyTable(['graph_type', 'rep', 'weight'])
        results = []
        for _ in range(num_iterations):
            # Create a different graph for each iteration
            graph = self.create_graph(loops, bin_size, window_size, random=True,
                                      num_loops=num_loops_to_choose)
            o_graph = o_chrom.create_graph(o_loops,bin_size, window_size,
                                           random=True, num_loops=num_loops_to_choose)
            result = self.get_stats(graph, o_graph, o_chrom, 'random',
                                    inner_table)
            results.append(result)

            value_table.add_row([x for x in list(result.values())])

        log_bin.info(value_table)

        # Take average of all iterations
        final_result = {
            'graph_type': 'random',
            'w': sum(x['w'] for x in results) / num_iterations,
            'rep': sum(x['rep'] for x in results) / num_iterations
        }

        log.debug(f'Random time: {time.time() - start_time} ------------------')
        return final_result

    def compare_subsample(self, loops, o_loops, o_chrom, bin_size,
                          window_size, inner_table):
        """
        Only compare top loops to better match samples with differing sequencing
         depth

        (Deprecated) (Doesn't work?)
        """

        start_time = time.time()

        num_loops_to_choose = min(len(loops), len(o_loops))

        # Sort the index list that has more to take the top loops
        if len(loops) > num_loops_to_choose:
            loops = loops.tolist()
            loops.sort(key=lambda x: self.value_list[x], reverse=True)
            log.debug(f"Taking the top {num_loops_to_choose} of "
                      f"{self.sample_name}")
        if len(o_loops) > num_loops_to_choose:
            o_loops = o_loops.tolist()
            o_loops.sort(key=lambda x: o_chrom.value_list[x], reverse=True)
            log.debug(f"Taking the top {num_loops_to_choose} of "
                      f"{o_chrom.sample_name}")

        value_table = PrettyTable(['graph_type', 'rep', 'weight'])
        graph = self.create_graph(loops, bin_size, window_size,
                                  num_loops=num_loops_to_choose)
        o_graph = o_chrom.create_graph(o_loops, bin_size, window_size,
                                       num_loops=num_loops_to_choose)
        result = self.get_stats(graph, o_graph, o_chrom, 'subsample',
                                inner_table)

        value_table.add_row([x for x in list(result.values())])
        log_bin.info(value_table)
        log.debug(f'Sub Sample time: {time.time() - start_time} --------------')

        return result

    def compare_only_match(self, graph, o_graph, o_chrom, inner_table,
                           graph1_smaller):
        """
        Only compare loops that match with other sample to better match
        differing sequencing depth

        (Deprecated) (Doesn't work?)
        """

        start_time = time.time()

        graph_type = ['mod', 'orig']

        # Returns [(mod1, orig1), (mod2, orig2)]
        new_graph_list = get_matching_graphs(graph, o_graph)
        value_table = PrettyTable(['graph_type', 'rep', 'weight'])

        # Compare only mod vs. orig
        for i in range(2):
            j = 1 - i

            if graph1_smaller and i == 0:
                continue

            new_graph = new_graph_list[0][i]
            new_o_graph = new_graph_list[1][j]

            result = self.get_stats(new_graph, new_o_graph, o_chrom,
                                    f'{graph_type[i]}_{graph_type[j]}',
                                    inner_table)

            value_table.add_row([x for x in list(result.values())])

        log_bin.info(value_table)
        log.debug(f'Match time: {time.time() - start_time} ------------------')

        # Not sure what to return: max, avg?
        return result

    def compare(self, o_chrom, window_start, window_end, bin_size, window_size,
                is_rep=False):
        """
        Compare a window of this chromosome to another chromosome from another
        sample

        Parameters
        ----------
        o_chrom : ChromLoopData
            The other chromosome
        window_start : int
            The start of the window
        window_end : int
            The end of the window
        bin_size : int
            Determines which loops are the same by putting them into bins
        window_size : int
            Needed to find the exact loop start/end index in this graph
        is_rep : bool, optional
            Debugging purposes

        Returns
        -------
        dict
            graph_type : str
                Information about which comparison method was used
            rep : str
                Reproducibility statistic for this window comparison
            w : str
                Weight of this window
        """

        # def compare(self, o_chrom, combined_peak_list, is_rep=False):
        start_time = time.time()

        if window_end > self.size:
            window_end = self.size

        if window_start >= self.size:
            log.error(f"Start of window ({window_start}) is larger than "
                      f"{self.name} size: {self.size}")
            return {'graph_type': 'error',
                    'j_value': 0,
                    'emd_value': 0,
                    'linear_emd_value': 0,
                    'emd_dist': 1,
                    'w': 0}

        log.debug(f'{self.sample_name} vs. {o_chrom.sample_name} '
                  f'{self.name}:{window_start} - {window_end}')

        # Get areas removed due to overlapping start/end anchors
        remove_time = time.time()
        combined_removed = get_removed_area(self.size, self, o_chrom)
        log.debug(f'Removed Area time: {time.time() - remove_time}')

        # Get loop indexes in the window
        loop_time = time.time()
        loops = get_loops(window_start, window_end, combined_removed,
                          self.filtered_start, self.filtered_end,
                          self.filtered_values)
        o_loops = get_loops(window_start, window_end, combined_removed,
                            o_chrom.filtered_start, o_chrom.filtered_end,
                            o_chrom.filtered_values)
        # loops = get_loops(0, self.size, combined_removed,
        #                   self.filtered_start, self.filtered_end,
        #                   self.filtered_values)
        # o_loops = get_loops(0, self.size, combined_removed,
        #                     o_chrom.filtered_start, o_chrom.filtered_end,
        #                     o_chrom.filtered_values)
        log.debug(f'Get Loop time: {time.time() - loop_time}')

        num_loops = len(loops)
        num_o_loops = len(o_loops)
        log.debug(f"Numb of loops in {self.sample_name}: {num_loops}")
        log.debug(f"Numb of loops in {o_chrom.sample_name}: {num_o_loops}")

        # value_table = PrettyTable(['graph_type', 'rep', 'weight'])
        # inner_value_table = \
        #     PrettyTable(['graph_type', 'j_value', 'emd', 'weight'])

        result = None
        if num_loops == 0 and num_o_loops == 0:
            result = {
                'graph_type': 'pass_none',
                'j_divergence': 0,
                'j_value': 1,
                'emd_value': 1,
                'emd_dist': 0,
                'w': 0
            }
            log.debug('No loops in either sample')

        # Randomly subsample from larger sample to better match miseq vs. hiseq
        # result = self.compare_randomly_choose(loops, o_loops, o_chrom,
        #                                       bin_size, window_size,
        #                                       inner_value_table)
        # log_bin.info(inner_value_table)
        # inner_value_table.clear_rows()
        # all_results.append(result)
        # value_table.add_row([x for x in list(result.values())])

        # Subsample top loops from larger sample to better match miseq vs. hiseq
        # result = self.compare_subsample(loops, o_loops, o_chrom, bin_size,
        #                                 window_size, inner_value_table)
        # log_bin.info(inner_value_table)
        # inner_value_table.clear_rows()
        # all_results.append(result)
        # value_table.add_row([x for x in list(result.values())])

        # if greater_numb / lesser_numb < 8:
        if result is None:
            # Make graphs using all loops in the window
            graph = self.create_graph(loops, bin_size, window_size,
                                      to_debug=is_rep)
            o_graph = o_chrom.create_graph(o_loops, bin_size,
                                           window_size, to_debug=is_rep)
            # merged_list = merge_peaks(combined_peak_list)
            # graph = create_peak_graph(merged_list, loops, self)
            # o_graph = create_peak_graph(merged_list, o_loops, o_chrom)

            # if self.numb_values / o_chrom.numb_values > 5 or \
            #        o_chrom.numb_values / self.numb_values > 5:
            # Match graphs only where loops overlap
            # graph1_smaller = self.numb_values < o_chrom.numb_values
            #
            # match_time = time.time()
            # result = self.compare_only_match(graph, o_graph, o_chrom,
            #                                  inner_value_table,
            #                                  graph1_smaller)
            # log.debug(f'Match time: {time.time() - match_time} -----------')
            # log_bin.info(inner_value_table)
            # inner_value_table.clear_rows()
            # all_results.append(result)
            # value_table.add_row([x for x in list(result.values())])

            # Normal comparison
            # norm_time = time.time()
            result = self.get_stats(graph, o_graph, o_chrom, 'norm')
            # log.debug(f'Norm time: {time.time() - norm_time} -------------')
            # log_bin.info(inner_value_table)
            # inner_value_table.clear_rows()
            # value_table.add_row([x for x in list(result.values())])

        # log_bin.info(value_table)
        log.debug(f'Total time: {time.time() - start_time}')
        # if is_rep:
        log.debug(f'emd_dist: {result["emd_dist"]}')
        log.debug(f'emd_value: {result["emd_value"]}')
        log.debug(f'j_divergence: {result["emd_value"]}')
        log.debug(f'j_value: {result["j_value"]}')
        #     for i in [0.5, 0, -0.5]:
        #         if result['emd_value'] < i:
        #             log.debug(f'Less than {i}')

        log.debug(
            '-----------------------------------------------------------------')

        return result

    def find_diff_loops(self, o_chrom, window_start, window_end, bin_size,
                        window_size, combined_peak_list, total_samples,
                        o_total_samples, graph_arr):

        log.debug(f'{self.sample_name} vs. {o_chrom.sample_name} '
                  f'{self.name}:{window_start} - {window_end}')

        ratio_list = np.zeros(int(window_size / NORM_LEN), dtype=np.float64)
        norm_start_index = int(window_start / NORM_LEN)
        out_file = open('diff_loops/norm_arr.txt', 'w')
        test_file = open('diff_loops/test_norm_arr.txt', 'w')

        for i in range(int(window_size / NORM_LEN)):
            norm_index = norm_start_index + i
            if self.norm_list[norm_index] > MIN_PEAK_VALUE and o_chrom.norm_list[norm_index] > MIN_PEAK_VALUE:
                ratio_list[i] = np.log2(self.norm_list[norm_index] / o_chrom.norm_list[norm_index])
            else:
                ratio_list[i] = 0
            out_file.write(f'{norm_index * NORM_LEN}\t{ratio_list[i]}\n')
            test_file.write(f'{norm_index * NORM_LEN}\t'
                            f'{self.norm_list[norm_index]}\t'
                            f'{o_chrom.norm_list[norm_index]}\t'
                            f'{ratio_list[i]}\n')
        out_file.close()
        test_file.close()

        bin_ratio_list = []
        num_norm_in_bin = int(bin_size / NORM_LEN)
        for i in range(bin_size):
            start_index = i * bin_size
            values = [x for x in
                      ratio_list[start_index:start_index+num_norm_in_bin] if x != 0]
            if len(values) == 0:
                bin_ratio_list.append(0)
            else:
                bin_ratio_list.append(np.mean(values))

        graph_len = ceil(window_size / bin_size)

        # Around 6GB
        if graph_len * graph_len * 2 * 8 / 1000000000 > 8:
            log.error(f'{self.size / bin_size} is too many rows for a graph')
            return []

        combined_removed = get_removed_area(self.size, self, o_chrom)
        loops = get_loops(window_start, window_end, combined_removed,
                          self.filtered_start, self.filtered_end,
                          self.filtered_values)
        o_loops = get_loops(window_start, window_end, combined_removed,
                            o_chrom.filtered_start, o_chrom.filtered_end,
                            o_chrom.filtered_values)

        # peak_arr = np.zeros(self.size, dtype=bool)
        # for peak in combined_peak_list:
        #     peak_start = peak[PEAK_START]
        #     peak_end = peak[PEAK_END]
        #     peak_arr[peak_start:peak_end] = True

        graph = self.create_graph(loops, bin_size, window_size)
        o_graph = o_chrom.create_graph(o_loops, bin_size, window_size)

        graph /= np.sum(graph)
        o_graph /= np.sum(o_graph)

        if np.sum(graph) == 0 and np.sum(o_graph) == 0:
            return []

        diff_areas = []

        start_time = time.time()

        for i in range(graph_len):
            for j in range(graph_len):

                value = graph[i][j]
                other_value = o_graph[i][j]

                norm_factor = (bin_ratio_list[i] + bin_ratio_list[j]) / 2
                norm_factor = 0

                if norm_factor != 0:
                    other_value *= (2 ** norm_factor)

                # Strength in other graph
                value_diff = value - other_value
                # value_fc = np.log2(value / other_value)

                if value_diff > 0.0001:
                    graph_arr[0].append(value)
                    graph_arr[1].append(other_value)

                    diff_areas.append((window_start + bin_size * i,
                                       window_start + bin_size * j,
                                       value_diff))
        end_time = time.time()

        return diff_areas

        # merged_list = merge_peaks(combined_peak_list + self.filtered_anchors + o_chrom.filtered_anchors)
        #
        # graph_size = len(merged_list)
        # peak_graph = self.create_peak_graph(merged_list)
        # o_peak_graph = o_chrom.create_peak_graph(merged_list)
        #
        # peak_centers = []
        # for peak in merged_list:
        #     peak_centers.append(int((peak[PEAK_START] + peak[PEAK_END]) / 2))
        #
        # output_file_name = f'{self.sample_name}_{o_chrom.sample_name}'
        #
        # # Weigh the difference by its relation to the max possible difference
        # max_value = max(np.max(peak_graph), np.max(o_peak_graph))
        #
        # with open(f'{output_dir}/{output_file_name}.peaks', 'w') as out_file:
        #     for i in merged_list:
        #         out_file.write(f'{i}\n')
        #
        # with open(f'{output_dir}/{output_file_name}.loops', 'w') as out_file:
        #     # out_file.write(f'{max_value}\n')
        #     for i in range(graph_size):
        #         for j in range(graph_size):
        #             diff = abs(peak_graph[i][j] - o_peak_graph[i][j])
        #             comb = peak_graph[i][j] + o_peak_graph[i][j]
        #             if comb == 0:
        #                 continue
        #
        #             # out_file.write(f'{self.name}\t{peak_centers[i]}\t'
        #             #                f'{peak_centers[j]}\t{diff}\t{comb}\t{1 - diff / max_value}\n')
        #             out_file.write(f'{self.name}\t{peak_centers[i]}\t'
        #                            f'{peak_centers[j]}\t{1 - diff / max_value}\n')

    def create_diff_graph(self, loops, bin_size, window_size, peak_arr):
        graph_len = ceil(window_size / bin_size)

        # 2D array of arrays
        # (none, single, both) peak support
        graph = [[([], [], []) for _ in range(graph_len)] for _ in range(graph_len)]

        for loop_index in loops:

            value = self.filtered_values[loop_index]
            start = self.filtered_start[loop_index]
            end = self.filtered_end[loop_index]

            window_start = start % window_size
            window_end = end % window_size

            # loop_len[int((end - start) / self.bin_size)] += 1

            bin_start = int(window_start / bin_size)
            bin_end = int(window_end / bin_size)

            peak_supp = 0
            if peak_arr[start]:
                peak_supp += 1
            if peak_arr[end]:
                peak_supp += 1

            graph[bin_start][bin_end][peak_supp].append([start, end, value])

            # if bin_end < bin_start:
            #     log.error(
            #         f'{orig_start}\t{orig_end}\t{start}\t{end}\t{bin_start}\t{bin_end}')

            # Get the other side of the graph as well for emd calculation
            # graph[bin_end][bin_start] += value

            # Avoid double counting the middle
            # if bin_end != bin_start:
            #    graph[bin_end][bin_start] += value

            # Also get areas surrounding this loop
            # May not be needed with emd calculation
            # Helps with finding jensen-shannon
            # for j in range(bin_start - 1, bin_start + 2):
            #     if j < 0 or j == graph_len:
            #         continue
            #     for k in range(bin_end - 1, bin_end + 2):
            #         if k < 0 or k == graph_len:
            #             continue
            #         graph[j][k] += value
                    # graph[k][j] += value
                    # if j != k:
                    #    graph[k][j] += value

            # if to_debug:
            #     log.debug(
            #         f'{self.sample_name}\t{orig_start}\t{orig_end}\t{value}')

        for i in graph_len:
            for j in graph_len:
                for k in 3:
                    graph[i][j][k].sort(key=lambda l: l[0])

                    loop_bin = graph[i][j][k]
                    merged_list = []

                    # Merge all loops with the same peak
                    # Loops anchors already set to peak in preprocessing
                    for key, group in itertools.groupby(loop_bin, lambda l: l[0]):
                        total_sum = 0
                        loop_end = None
                        for arr in group:
                            total_sum += arr[2]
                            if not loop_end:
                                loop_end = arr[1]
                            if loop_end != arr[1]:
                                log.error("")

                        merged_list.append((key, group[0], sum))
                        curr_start = loop_bin[loop_index][0]
                        prev_start = loop_bin[loop_index - 1][0]

                        if curr_start == prev_start:
                            continue



        # log.info(f"Number of loops in {self.sample_name} graph: {num_loops_used}")

        # plt.plot([x for x in range(len(loop_len))], [np.log(x) for x in loop_len])
        # plt.show()

        # log.info(f'Max value in graph: {np.max(graph)}')
        # return graph, total_PET_count / self.total_loop_value
        return graph

    # Weight each
    def create_peak_graph(self, merged_list):
        graph_len = len(merged_list)
        graph = np.zeros((graph_len, graph_len), dtype=np.float64)

        peak_array = np.full(self.size, -1, dtype=np.int16)
        for i, peak in enumerate(merged_list):
            start = peak[PEAK_START_INDEX]
            end = peak[PEAK_END_INDEX]
            peak_array[start:end] = i

        log.info(f'Numb of loops: {self.filtered_numb_values}')
        for loop_index in range(self.filtered_numb_values):
            value = self.filtered_values[loop_index]
            start = self.filtered_start[loop_index]
            end = self.filtered_end[loop_index]

            start_index = peak_array[start]
            end_index = peak_array[end]

            if start_index == -1 or end_index == -1:
                log.critical(f"Loop: ({start}, {end}, {value}) "
                             f"does not have a corresponding peak")
                continue

            graph[start_index][end_index] += value

        return graph / graph.sum()


def merge_peaks(combined_peak_list):
    """
    Merges overlapping peaks
    """

    found = False
    for interval in combined_peak_list:
        if interval[0] == 78001026:
            found = True
            break
    assert found

    combined_peak_list.sort(key=lambda x: x[PEAK_START_INDEX])
    merged_peak_list = []
    peak_diffs = []
    for higher_peak in combined_peak_list:
        if not merged_peak_list:
            merged_peak_list.append(higher_peak)
            continue

        lower_peak = merged_peak_list[-1]

        print()
        print(lower_peak)
        print(higher_peak)

        # "higher" peak is entirely inside the lower peak
        if higher_peak[PEAK_START_INDEX] <= lower_peak[PEAK_END_INDEX] and \
                lower_peak[PEAK_END_INDEX] >= higher_peak[PEAK_END_INDEX]:
            # lower_peak[PEAK_MAX_VALUE] += higher_peak[PEAK_MAX_VALUE]

            peak_diffs.append(1)
            print('Entirely inside')
            continue

        # Find percentage of peak area that overlaps
        dist_diff = higher_peak[PEAK_START_INDEX] - lower_peak[PEAK_END_INDEX]
        # lower_peak_percentage = dist_diff / lower_peak[PEAK_LEN]
        # higher_peak_percentage = dist_diff / higher_peak[PEAK_LEN]

        # Has to have at least 50% overlapping for one peaks
        # if lower_peak_percentage >= 0.5 or higher_peak_percentage >= 0.5:
        if dist_diff < 5000:
            # peak_diffs.append(
            #     max(lower_peak_percentage, higher_peak_percentage))

            lower_peak[PEAK_END_INDEX] = higher_peak[PEAK_END_INDEX]
            # lower_peak[PEAK_MAX_VALUE] += higher_peak[PEAK_MAX_VALUE]
            lower_peak[PEAK_LEN_INDEX] = lower_peak[PEAK_END_INDEX] - lower_peak[
                PEAK_START_INDEX]
            print('Less than 5k away')
            continue

        merged_peak_list.append(higher_peak)
        print('New peak')

    log.info(f"Merged {len(combined_peak_list) - len(merged_peak_list)} peaks")
    log.info(f"Avg non-overlapping percentage between merged peaks: "
             f"{np.mean(peak_diffs)}")
    log.info(f'Numb of peaks left: {len(merged_peak_list)}')

    return merged_peak_list
