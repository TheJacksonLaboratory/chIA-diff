import logging
import time

import numpy as np
import os
import itertools
from .cython_util import *
from .util import *
from math import ceil, sqrt
import matplotlib.pyplot as plt
import random
import json
import prettytable

PEAK_START_INDEX = 0
PEAK_END_INDEX = 1
PEAK_LEN_INDEX = 2
PEAK_MAX_VALUE_INDEX = 3
PEAK_MEAN_VALUE_INDEX = 4

NODE_WEIGHT_INDEX = -1
CONTINUE_PROB_INDEX = 2

COVERAGE_BIN_SIZE = 500

VERSION = 8

log = logging.getLogger()
random.seed(0)
np.random.seed(0)


def get_skip_area(chrom_size, chrom_list):
    """
    Gets skipped area from both samples so no loops from that area are compared

    Parameters
    ----------
    chrom_size : int
        Size of chromosome
    chrom_list : list of ChromLoopData

    Returns
    -------
    1D Numpy array, uint8
        Array marked with 1 in removed intervals
    """

    starts = []
    ends = []

    # Combine the removed areas of each chromosome given
    for chrom in chrom_list:
        starts.extend(chrom.removed_intervals[0])
        ends.extend(chrom.removed_intervals[1])

    to_remove = np.array([starts, ends], dtype=np.int32)

    # Uses dtype unsigned char because bool doesn't work as well in Cython
    removed_area = np.zeros(chrom_size, dtype=np.uint8)
    for i in range(len(to_remove[0])):
        start = to_remove[0][i]
        end = to_remove[1][i]
        removed_area[start:end] = 1

    return removed_area


def create_graph_adj_list(chrom, loop_indexes, loop_bin_size, window_start,
                          window_end):
    """
    Creates an adjacency list representation of a directed graph. Nodes are
    non-overlapping bins inside the specified window that contain loops. Edges
    are loops that start from one bin to another, weighted by the coverage
    amount in the anchors.

    Parameters
    ----------
    chrom : ChromLoopData
    loop_indexes : list of int
        Holds a list of indexes to the main loop lists in chrom
    loop_bin_size : int
        Tested to work best with 5kb
    window_start : int
    window_end : int

    Returns
    -------
    dict
        key: bin number
        value: [list of keys from bin (edges), list of normalized weights for
                each edge, max edge value, weight of bin]
    """

    # Should probably be this way
    if loop_bin_size % COVERAGE_BIN_SIZE != 0:
        error_msg = f'Bin size: {loop_bin_size} must be a multiple of ' \
                    f'PEAK_BIN_LEN: {COVERAGE_BIN_SIZE}'
        log.error(error_msg)
        raise ValueError(error_msg)

    node_dict = {}

    for loop_index in loop_indexes:
        pet_count = chrom.pet_count_list[loop_index]
        start = chrom.start_list[loop_index]
        end = chrom.end_list[loop_index]

        assert end < window_end
        assert start > window_start

        # Round down
        loop_bin_num = int((start - window_start) / loop_bin_size)
        bin_end = int((end - window_start) / loop_bin_size)

        # Filter out loops that start and end in the same bin
        if loop_bin_num == bin_end:
            continue

        # Weight each loop by the front and back anchor
        forward_value = pet_count * chrom.start_anchor_weights[loop_index]
        back_value = pet_count * chrom.end_anchor_weights[loop_index]

        # Only include the loop if the weighted value is not 0
        # Loop starts from back anchor
        if back_value != 0:
            if bin_end not in node_dict:
                node_dict[bin_end] = {}

            if loop_bin_num not in node_dict[bin_end]:
                node_dict[bin_end][loop_bin_num] = 0

            node_dict[bin_end][loop_bin_num] += back_value

        # Loop starts from front anchor
        if forward_value != 0:
            if loop_bin_num not in node_dict:
                node_dict[loop_bin_num] = {}

            if bin_end not in node_dict[loop_bin_num]:
                node_dict[loop_bin_num][bin_end] = 0

            node_dict[loop_bin_num][bin_end] += forward_value

    num_coverage_in_loop_bin = int(loop_bin_size / COVERAGE_BIN_SIZE)
    coverage_bin_start = int(window_start / COVERAGE_BIN_SIZE)
    coverage_bin_end = ceil(window_end / COVERAGE_BIN_SIZE)
    normalized_coverage_bins = \
        np.array(chrom.coverage_bins[coverage_bin_start:coverage_bin_end])
    normalized_coverage_bins /= np.sum(normalized_coverage_bins)
    assert len(normalized_coverage_bins) == ceil(
        (window_end - window_start) / COVERAGE_BIN_SIZE)
    assert len(normalized_coverage_bins) == ceil(
        (window_end - window_start) / loop_bin_size *
        (loop_bin_size / COVERAGE_BIN_SIZE))

    loop_bin_weights = []
    loop_bin_weight = 0
    for i in range(len(normalized_coverage_bins)):
        loop_bin_weight += normalized_coverage_bins[i]

        # If first 10 coverage bins are wanted, get bins 0-9 inclusive
        if (i + 1) % num_coverage_in_loop_bin == 0:
            loop_bin_weights.append(loop_bin_weight)
            loop_bin_weight = 0

    # Make sure all bin weights were used
    assert loop_bin_weight == 0

    node_degrees = []

    # Max value for a loop inside a bin
    max_bin_loop_value = 0
    for loop_bin_num, node in node_dict.items():
        neighbors = np.array(list(node.keys()))
        edge_values = np.array(list(node.values()))

        node_degrees.append(len(neighbors))

        max_edge_value = np.max(edge_values)
        if max_edge_value > max_bin_loop_value:
            max_bin_loop_value = max_edge_value

        node_dict[loop_bin_num] = [neighbors, edge_values / np.sum(edge_values),
                                   max_edge_value, loop_bin_weights[loop_bin_num]]

    log.info(f'Max loop value in this window: {max_bin_loop_value}')

    for loop_bin_num in node_dict:
        node_dict[loop_bin_num][CONTINUE_PROB_INDEX] = \
            node_dict[loop_bin_num][CONTINUE_PROB_INDEX] / max_bin_loop_value

    # Debugging purposes
    # bin_numb = int(815000 / loop_bin_size)
    # if window_start == 0 and bin_numb in node_dict:
    #     print(node_dict[bin_numb])

    # Find percentage of nodes that have n degrees
    # total_nodes = len(node_dict)
    # total_potential_nodes = ceil((window_end - window_start) / loop_bin_size)
    # log.info(f'{(total_potential_nodes - total_nodes) / total_potential_nodes * 100}% of '
    #          f'bins have zero loops attached')
    # for i in range(1, 10):
    #     log.info(
    #         f'{len([x for x in node_degrees if x == i]) / total_potential_nodes * 100}% of '
    #         f'bins have {i} loops attached')

    return node_dict


def create_graph_matrix(chrom, loops, bin_size, window_start, to_debug=False,
                        output_dir=None):
    """
    Creates a bin-based graph to easily compare loops

    Parameters
    ----------
    chrom : ChromLoopData
    loops : list
        List of loop indexes in this window
    bin_size : int
    to_debug : bool, optional
        Log loops used in the graph (Default is False)
    output_dir : str, optional
        Directory to output graph (Default is 'graph_txt')

    Returns
    -------
    Numpy 2D array
    """

    graph_len = ceil(chrom.window_size / bin_size)
    graph = np.ones((graph_len, graph_len), dtype=np.float64)

    for loop_index in loops:
        value = chrom.pet_count_list[loop_index]
        start = chrom.start_list[loop_index]
        end = chrom.end_list[loop_index]

        start = start % chrom.window_size
        end = end % chrom.window_size

        bin_start = int(start / bin_size)
        bin_end = int(end / bin_size)

        graph[bin_start][bin_end] += value * chrom.start_anchor_weights[
            loop_index]
        graph[bin_end][bin_start] += value * chrom.end_anchor_weights[
            loop_index]

        # Also get areas surrounding this loop
        # for j in range(bin_start - 1, bin_start + 2):
        #     if j < 0 or j == graph_len:
        #         continue
        #     for k in range(bin_end - 1, bin_end + 2):
        #         if k < 0 or k == graph_len:
        #             continue
        #         graph[j][k] += value * self.start_anchor_weights[i]
        #         graph[k][j] += value * self.end_anchor_weights[i]

    # log.info(f'Max value in graph: {np.max(graph)}')
    # return graph, total_PET_count / self.total_loop_value

    if output_dir:
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        output_path = f'{output_dir}/{chrom.sample_name}_{window_start}.' \
                      f'window_size={chrom.window_size}.' \
                      f'bin_size={bin_size}.txt'
        np.savetxt(output_path, graph, '%05.5f', '\t')

    return graph


def find_diff_loops(sample1, sample2, bin_size, window_index=None,
                    chroms_to_diff=None, start_index=None, end_index=None,
                    output_dir=None):
    if output_dir:
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

    if chroms_to_diff is None:
        chroms_to_diff = list(sample1.chrom_dict.keys())
    log.info(f'Chroms to compare: {chroms_to_diff}')

    for chrom_name in chroms_to_diff:

        if chrom_name not in sample1.chrom_dict:
            log.warning(f'{chrom_name} is not in {sample1.sample_name}. '
                        f'Skipping {chrom_name}')
            continue

        if chrom_name not in sample2.chrom_dict:
            log.warning(f'{chrom_name} is in {sample1.sample_name} but '
                        f'not in {sample2.sample_name}. Skipping '
                        f'{chrom_name}')
            continue

        log.info(f"Finding different loops for {chrom_name} ...")

        chrom1 = sample1.chrom_dict[chrom_name]
        chrom2 = sample2.chrom_dict[chrom_name]

        chrom_loop_ratios = []
        o_chrom_loop_ratios = []
        loop_count_refs = []
        for i in range(len(chrom1.window_total_loop_count)):
            loop_count_ref = sqrt(
                chrom1.window_total_loop_count[i] *
                chrom2.window_total_loop_count[i])

            loop_count_refs.append(loop_count_ref)

            if loop_count_ref == 0:
                loop_count_ref = 1

            chrom_loop_ratios.append(
                chrom1.window_total_loop_count[i] / loop_count_ref)
            o_chrom_loop_ratios.append(
                chrom2.window_total_loop_count[i] / loop_count_ref)

        chrom_loop_median = np.median(chrom_loop_ratios)
        o_chrom_loop_median = np.median(o_chrom_loop_ratios)

        log.info(f'Norm factor for {sample1.sample_name}: {chrom_loop_median}')
        log.info(f'Norm factor for {sample2.sample_name}: '
                 f'{o_chrom_loop_median}')

        # Compare for all windows in chrom
        chrom_size = chrom1.size

        # TODO: Bug here when comparing multiple chromosomes if
        #  start_index was None
        if start_index is None:
            start_index = 0
        if end_index is None:
            end_index = chrom_size

        if end_index < start_index:
            log.error(f"Given interval: ({start_index}, {end_index}) to "
                      f"diff. Cannot compute when start is greater than end.")
            return

        numb_windows = ceil((end_index - start_index) / sample1.window_size)

        diff_areas = []
        graph_arr = ([], [], [], [])

        output_file_name = f'{sample1.sample_name}_{sample2.sample_name}'
        for k in range(numb_windows):

            # If there is a specified window, just compare that
            if window_index is not None:
                k = window_index

            window_start = sample1.window_size * k + start_index
            window_end = window_start + sample1.window_size
            if window_end > end_index:
                window_end = end_index

            diff_areas += \
                diff_chrom_window(chrom1, chrom2, window_start, window_end,
                                  bin_size, chrom_loop_median,
                                  o_chrom_loop_median,
                                  sample1.total_samples,
                                  sample2.total_samples, graph_arr)

            if window_index is not None:
                break

        log.info(f'Plotting scatter plot with {len(graph_arr[0])} points')
        plt.scatter(graph_arr[0], graph_arr[1])
        plt.title(f'{output_file_name}_chr1')
        plt.xlabel(f'{sample1.sample_name}')
        plt.ylabel(f'{sample2.sample_name}')
        plt.savefig(f'diff_loops/{output_file_name}_values')
        plt.close()

        diff_areas.sort(key=lambda l: l[2], reverse=True)
        if output_dir:
            with open(f'{output_dir}/{output_file_name}.loops',
                      'w') as out_file:
                for area in diff_areas:
                    out_file.write(f'{area[0]}\t{area[1]}\t{area[2]}\n')


def diff_chrom_window(chrom1, chrom2, window_start, window_end, bin_size,
                      norm_factor, o_norm_factor, total_samples,
                      o_total_samples, graph_arr):
    log.debug(f'{chrom1.sample_name} vs. {chrom2.sample_name} '
              f'{chrom1.name}:{window_start} - {window_end}')

    # ratio_list = np.zeros(int(window_size / NORM_LEN), dtype=np.float64)
    # norm_start_index = int(window_start / NORM_LEN)
    # out_file = open('diff_loops/norm_arr.txt', 'w')
    # test_file = open('diff_loops/test_norm_arr.txt', 'w')
    #
    # for i in range(int(window_size / NORM_LEN)):
    #     norm_index = norm_start_index + i
    #     if self.norm_list[norm_index] > MIN_PEAK_VALUE and o_chrom.norm_list[norm_index] > MIN_PEAK_VALUE:
    #         ratio_list[i] = np.log2(self.norm_list[norm_index] / o_chrom.norm_list[norm_index])
    #     else:
    #         ratio_list[i] = 0
    #     out_file.write(f'{norm_index * NORM_LEN}\t{ratio_list[i]}\n')
    #     test_file.write(f'{norm_index * NORM_LEN}\t'
    #                     f'{self.norm_list[norm_index]}\t'
    #                     f'{o_chrom.norm_list[norm_index]}\t'
    #                     f'{ratio_list[i]}\n')
    # out_file.close()
    # test_file.close()
    #
    # bin_ratio_list = []
    # num_norm_in_bin = int(bin_size / NORM_LEN)
    # for i in range(bin_size):
    #     start_index = i * bin_size
    #     values = [x for x in
    #               ratio_list[start_index:start_index+num_norm_in_bin] if x != 0]
    #     if len(values) == 0:
    #         bin_ratio_list.append(0)
    #     else:
    #         bin_ratio_list.append(np.mean(values))

    graph_len = ceil(chrom1.window_size / bin_size)

    # Around 6GB
    if graph_len * graph_len * 2 * 8 / 1000000000 > 8:
        log.error(f'{chrom1.size / bin_size} is too many rows for a graph')
        return []

    combined_removed = get_skip_area(chrom1.size, [chrom1, chrom2])

    # Get loops again in case a specific window_start/end was given
    loop_indexes = get_loops(window_start, window_end, combined_removed,
                             chrom1.start_list, chrom1.end_list,
                             chrom1.pet_count_list)
    o_loop_indexes = get_loops(window_start, window_end, combined_removed,
                               chrom2.start_list, chrom2.end_list,
                               chrom2.pet_count_list)

    graph = create_graph_matrix(chrom1, loop_indexes, bin_size, window_start)
    o_graph = create_graph_matrix(chrom2, o_loop_indexes, bin_size,
                                  window_start)

    graph /= norm_factor
    o_graph /= o_norm_factor

    diff_areas = []

    for i in range(graph_len):
        for j in range(graph_len):

            value = graph[i][j]
            other_value = o_graph[i][j]

            # norm_factor = (bin_ratio_list[i] + bin_ratio_list[j]) / 2
            # norm_factor = 0

            # if norm_factor != 0:
            #     other_value *= (2 ** norm_factor)

            # Strength in other graph
            value_diff = value - other_value
            # value_fc = np.log2(value / other_value)

            if abs(value_diff) > 100:
                graph_arr[0].append(value)
                graph_arr[1].append(other_value)

                diff_areas.append((window_start + bin_size * i,
                                   window_start + bin_size * j,
                                   value_diff))
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


def walk_graph(start_nodes, adj_list_dict, window_start, loop_bin_size,
               path_list, path_popularity, is_null_edge, all_node_list=None):
    """
    Walk through the given graph and keep track of the path and edge
    popularities

    Parameters
    ----------
    start_nodes : list
    adj_list_dict
    window_start
    loop_bin_size
    path_list
    path_popularity
    is_null_edge
    all_node_list : list

    """

    if is_null_edge:
        assert all_node_list is not None
        numb_all_nodes = len(all_node_list)
        assert numb_all_nodes > 1

    for i, start_bin_numb in enumerate(start_nodes):
        curr_bin_numb = start_bin_numb
        path = [start_bin_numb]

        if start_bin_numb not in adj_list_dict:
            path_list.append(path)
            continue

        curr_node = adj_list_dict[start_bin_numb]

        while True:
            # Continue to next node if there are large loops in this node
            if not np.random.binomial(1, curr_node[CONTINUE_PROB_INDEX]):
                break

            if is_null_edge:
                # Randomly choose a node from all potential nodes
                while True:
                    # Supposedly faster to do it this way
                    next_index = np.random.randint(0, numb_all_nodes)
                    next_bin_numb = all_node_list[next_index]

                    # Make sure not to choose current node
                    if next_bin_numb != curr_bin_numb:
                        break

                # next_bin_numb = np.random.choice(all_node_list)
            else:
                # Biggest time bottleneck... ~80% of execution time
                next_bin_numb = np.random.choice(curr_node[0], p=curr_node[1])

            path.append(next_bin_numb)

            curr_path = f'{window_start + loop_bin_size * curr_bin_numb}-' \
                        f'{window_start + loop_bin_size * next_bin_numb}'

            if curr_path not in path_popularity:
                path_popularity[curr_path] = 0
            path_popularity[curr_path] += 1

            # Random walk reached a bin where there were no loops exiting it
            if next_bin_numb not in adj_list_dict:
                break

            curr_node = adj_list_dict[next_bin_numb]
            curr_bin_numb = next_bin_numb

        path_list.append(path)


def random_walk(chrom, window_start=0, window_end=None, loop_bin_size=5000,
                walk_iter=50000):
    """
    Runs random walk(s) for the chromosome in the given window.

    Parameters
    ----------
    chrom : ChromLoopData
        The chromosome to random walk in
    window_start : int, optional
        The start of window to random walk in. (Default is 0)
    window_end : int, optional
        The end of window to random walk in. (Default is chrom.size)
    loop_bin_size : int, optional
        The bin size to group loops together. A smaller bin size might find
        false positive differential loops. A larger bin size might miss some
        differential loops. (Default is 5000)
    walk_iter : int, optional
        The number of iterations of random walks. (Default is 50000)

    Returns
    -------
    tuple, (2D list, dict)
        normal path list, normal path popularity
        random edge path list, random edge path popularity
        random edge/node path list, random edge/node path popularity
    """

    if window_end is None:
        window_end = chrom.size

    if not os.path.isdir('random_walks'):
        os.mkdir('random_walks')

    log.debug(f'Random walking on '
              f'{chrom.sample_name}.{chrom.name}:{window_start}-{window_end} '
              f'walk_iter={walk_iter} '
              f'loop_bin_size={loop_bin_size}')

    skip_area = get_skip_area(chrom.size, [chrom])
    loop_indexes = get_loops(window_start, window_end, skip_area,
                             chrom.start_list, chrom.end_list,
                             chrom.pet_count_list)
    if len(loop_indexes) == 0:
        log.warning(f'No loops found in '
                    f'{chrom.name}:{window_start}-{window_end}')
        return None

    adj_list_dict = create_graph_adj_list(chrom, loop_indexes, loop_bin_size,
                                          window_start, window_end)

    unused_loop_bins = []
    for bin_numb in range(ceil((window_end - window_start) / loop_bin_size)):
        if bin_numb not in adj_list_dict:
            unused_loop_bins.append(bin_numb)

    # Get the list of node weights and normalize them
    used_loop_bins = list(adj_list_dict.keys())
    node_weights = [adj_list_dict[x][NODE_WEIGHT_INDEX] for x in used_loop_bins]
    if np.sum(node_weights) == 0:
        log.warning(f'No coverage in {chrom.name}:{window_start}-{window_end}')
        return None
    node_weights /= np.sum(node_weights)

    output_node_weights(chrom, window_start, window_end, used_loop_bins,
                        walk_iter, node_weights)
    log.debug(f'Numb used loop bins: {len(used_loop_bins)}')
    log.debug(f'Numb unused loop bins: {len(unused_loop_bins)}')

    normal_path_list = []
    normal_popularity = {}

    edge_null_path_list = []
    edge_null_popularity = {}

    all_null_path_list = []
    all_null_popularity = {}

    total_node_list = unused_loop_bins + used_loop_bins

    # Pre-compute the starting nodes for the random walk
    start_nodes = np.random.choice(used_loop_bins, size=walk_iter,
                                   p=node_weights)
    null_start_nodes = np.random.choice(total_node_list, size=walk_iter)

    output_path_start_counts(chrom, window_start, window_end, loop_bin_size,
                             walk_iter, start_nodes, 'normal')
    output_path_start_counts(chrom, window_start, window_end, loop_bin_size,
                             walk_iter, null_start_nodes, 'null_dist')

    # The normal walk
    walk_graph(start_nodes, adj_list_dict, window_start, loop_bin_size,
               normal_path_list, normal_popularity, False)

    # The walk with random edges
    walk_graph(start_nodes, adj_list_dict, window_start, loop_bin_size,
               edge_null_path_list, edge_null_popularity, True, total_node_list)

    # The walk with random starts and random edges
    walk_graph(null_start_nodes, adj_list_dict, window_start, loop_bin_size,
               all_null_path_list, all_null_popularity, True, total_node_list)

    return (
        (normal_path_list, normal_popularity),
        (edge_null_path_list, edge_null_popularity),
        (all_null_path_list, all_null_popularity)
    )
