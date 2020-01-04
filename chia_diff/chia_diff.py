import logging
import time

import numpy as np
import os
import itertools
from .cython_util import *
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

BIN_WEIGHT_INDEX = -1

PEAK_BIN_LEN = 500

VERSION = 8

log = logging.getLogger()
random.seed(0)
np.random.seed(0)


def get_skipped_area(chrom_size, chrom_list):
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


def create_graph_adj_list(chrom, loop_indexes, bin_size, window_start,
                          window_end):
    node_dict = {}

    for loop_index in loop_indexes:
        pet_count = chrom.pet_count_list[loop_index]
        start = chrom.start_list[loop_index]
        end = chrom.end_list[loop_index]

        assert end < window_end
        assert start > window_start

        # Round down
        bin_start = int((start - window_start) / bin_size)
        bin_end = int((end - window_start) / bin_size)

        # Filter out loops that start and end in the same bin
        if bin_start == bin_end:
            continue

        forward_value = pet_count * chrom.start_anchor_weights[loop_index]
        back_value = pet_count * chrom.end_anchor_weights[loop_index]

        if back_value != 0:
            if bin_end not in node_dict:
                node_dict[bin_end] = {}

            if bin_start not in node_dict[bin_end]:
                node_dict[bin_end][bin_start] = 0

            node_dict[bin_end][bin_start] += back_value

        if forward_value != 0:
            if bin_start not in node_dict:
                node_dict[bin_start] = {}

            if bin_end not in node_dict[bin_start]:
                node_dict[bin_start][bin_end] = 0

            node_dict[bin_start][bin_end] += forward_value

    if bin_size % PEAK_BIN_LEN != 0:
        error_msg = f'Bin size: {bin_size} must be a multiple of ' \
                    f'PEAK_BIN_LEN: {PEAK_BIN_LEN}'
        log.error(error_msg)
        raise ValueError(error_msg)

    bin_weights = []
    bin_weight = 0
    numb_peak_bins = int(bin_size / PEAK_BIN_LEN)
    peak_start = int(window_start / PEAK_BIN_LEN)
    peak_end = ceil(window_end / PEAK_BIN_LEN)
    normalized_peak_bins = np.array(chrom.peak_bins[peak_start:peak_end])
    normalized_peak_bins /= np.sum(normalized_peak_bins)
    assert len(normalized_peak_bins) == ceil(
        (window_end - window_start) / PEAK_BIN_LEN)
    assert len(normalized_peak_bins) == ceil(
        (window_end - window_start) / bin_size * (bin_size / PEAK_BIN_LEN))

    for i in range(len(normalized_peak_bins)):
        bin_weight += normalized_peak_bins[i]
        if (i + 1) % numb_peak_bins == 0:
            bin_weights.append(bin_weight)
            bin_weight = 0

    assert bin_weight == 0

    # log.debug(f'Bin Weights: {bin_weights}')

    total_nodes = len(node_dict)
    total_potential_nodes = ceil((window_end - window_start) / bin_size)
    # log.info(f'{(total_potential_nodes - total_nodes) / total_potential_nodes * 100}% of '
    #          f'bins have zero loops attached')

    node_degrees = []
    max_loop_bin_value = 0
    for bin_start, node in node_dict.items():
        bin_ends = np.array(list(node.keys()))
        loops = np.array(list(node.values()))

        node_degrees.append(len(bin_ends))

        if np.max(loops) > max_loop_bin_value:
            max_loop_bin_value = np.max(loops)

        node_dict[bin_start] = [bin_ends, loops / np.sum(loops), np.max(loops),
                                bin_weights[bin_start]]

    log.info(f'Max value in loop bin: {max_loop_bin_value}')
    for bin_start in node_dict:
        node_dict[bin_start][2] = node_dict[bin_start][2] / max_loop_bin_value

    bin_numb = int(815000 / bin_size)
    if window_start == 0 and bin_numb in node_dict:
        print(node_dict[bin_numb])

    # for i in range(1, 10):
    #     log.info(
    #         f'{len([x for x in node_degrees if x == i]) / total_potential_nodes * 100}% of '
    #         f'bins have {i} loops attached')

    # bins = [i for i in range(9)] + [max(node_degrees)]
    # hist, bin_edges = np.histogram(node_degrees, bins)
    # print(hist)

    return node_dict, max_loop_bin_value


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

    combined_removed = get_skipped_area(chrom1.size, [chrom1, chrom2])

    # Get loops again in case a specific window_start/end was given
    loop_indexes = get_loops(window_start, window_end, combined_removed,
                             chrom1.start_list, chrom1.end_list,
                             chrom1.pet_count_list)
    o_loop_indexes = get_loops(window_start, window_end, combined_removed,
                               chrom2.start_list, chrom2.end_list,
                               chrom2.pet_count_list)

    # peak_arr = np.zeros(self.size, dtype=bool)
    # for peak in combined_peak_list:
    #     peak_start = peak[PEAK_START]
    #     peak_end = peak[PEAK_END]
    #     peak_arr[peak_start:peak_end] = True

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


def walk_graph(start_nodes, adj_list_dict, walk_len, window_start, bin_size,
               path_list, bin_popularity, is_null_edge, max_node_weight,
               max_loop_bin_value, all_node_list=None):
    """
    Walk through the given graph and keep track of the

    """
    test_count = 0

    if is_null_edge:
        assert all_node_list
        numb_all_nodes = len(all_node_list)
        assert numb_all_nodes > 1

    for i, start_bin in enumerate(start_nodes):
        curr_bin_numb = start_bin

        if start_bin not in adj_list_dict:
            path_list.append(start_bin)
            continue

        curr_node = adj_list_dict[start_bin]
        path = [start_bin]

        # for j in range(walk_len - 1):
        while True:

            if not np.random.binomial(1, curr_node[2]):
                break

            if is_null_edge:
                # Supposedly faster to do it this way
                while True:
                    next_index = np.random.randint(0, numb_all_nodes)
                    next_bin_numb = all_node_list[next_index]

                    if next_bin_numb != curr_bin_numb:
                        break

                # next_bin_numb = np.random.choice(all_node_list)
            else:
                next_bin_numb = np.random.choice(curr_node[0], p=curr_node[1])

            curr_path = f'{window_start + bin_size * curr_bin_numb}-' \
                        f'{window_start + bin_size * next_bin_numb}'
            path.append(next_bin_numb)

            if curr_path not in bin_popularity:
                bin_popularity[curr_path] = 0
            bin_popularity[curr_path] += 1

            # if curr_path == '2050000-2060000' and window_start == 0:
            #     print(i, j, path)
            #     test_count += 1

            if next_bin_numb not in adj_list_dict:
                break

                # TODO: Should the random walk get stuck? This is because of a
                #  loop ending in a place with no supporting bedgraph
                log.error("Error with adj list graph")
                raise RuntimeError(
                    f"Bug with chia_diff: {next_bin_numb} is not a "
                    f"valid bin")

            curr_node = adj_list_dict[next_bin_numb]
            curr_bin_numb = next_bin_numb

        # path_len = len(path)
        # path.append(path_len)

        # if (start_bin == 410 or start_bin == 427) and window_start == 0:
        #     # test_count += 1
        #     print(i, path)

        path_list.append(path)

    # print(test_count)


def random_walk(chrom, loop_bin_size, walk_len=3, walk_iter=50000,
                window_start=0, window_end=None):
    """
    Runs random walk(s) for the chromosome in the given window

    Parameters
    ----------
    chrom : ChromLoopData
        The chromosome to random walk in
    loop_bin_size : int
        The bin size to group loops together. A smaller bin size might find
        false positive differential loops. A larger bin size might miss some
        differential loops.
    walk_len : int, optional
        The length of each walk for each iteration of a random walk.
        (Default is 50)
    walk_iter : int, optional
        The number of iterations of random walks. (Default is 5000)
    window_start : int, optional
        The start of window to random walk in. (Default is 0)
    window_end : int, optional
        The end of window to random walk in. (Default is chrom.size)
    """

    if window_end is None:
        window_end = chrom.size

    log.debug(f'Random walking on {chrom.sample_name}: {walk_len} {walk_iter} '
              f'{window_start}-{window_end}')

    combined_removed = get_skipped_area(chrom.size, [chrom])

    # Get loops again in case a specific window_start/end was given
    loop_indexes = get_loops(window_start, window_end, combined_removed,
                             chrom.start_list, chrom.end_list,
                             chrom.pet_count_list)
    if len(loop_indexes) == 0:
        log.warning(f'No loops in {window_start}-{window_end}')
        return None

    # Create the graph adjacency list
    adj_list_dict, max_loop_bin_value = \
        create_graph_adj_list(chrom, loop_indexes, loop_bin_size, window_start,
                              window_end)

    unattached_bins = []
    for bin_numb in range(ceil((window_end - window_start) / loop_bin_size)):
        if bin_numb not in adj_list_dict:
            unattached_bins.append(bin_numb)

    # Get the list of node weights and normalize them
    node_list = list(adj_list_dict.keys())
    node_weights = [adj_list_dict[node][BIN_WEIGHT_INDEX] for node in node_list]

    total_potential_nodes = unattached_bins + node_list

    if np.sum(node_weights) == 0:
        log.warning(f'No bedgraph values in {window_start}-{window_end}')
        log.warning([window_start + loop_bin_size * n for n in node_list])
        return None

    node_weights /= np.sum(node_weights)

    # log.debug(f'Node list len: {len(node_list)}')
    # log.debug(f'Node weights len: {len(node_weights)}')

    with open(f'random_walks/{chrom.sample_name}.{chrom.name}.'
              f'{window_start}-{window_end}.node_weights.txt', 'w') as out_file:
        for i in range(len(node_list)):
            out_file.write(f'{node_list[i]}\t{node_weights[i]}\n')

    path_list = []
    bin_popularity = {}

    edge_null_walks = []
    edge_null_popularity = {}

    all_null_walks = []
    all_null_popularity = {}

    start_nodes = np.random.choice(node_list, size=walk_iter, p=node_weights)
    null_start_nodes = np.random.choice(total_potential_nodes, size=walk_iter)

    with open(f'random_walks/{chrom.sample_name}.{chrom.name}.'
              f'{window_start}-{window_end}.normal_start_node.txt',
              'w') as out_file:
        node_popularity = {}
        for node in start_nodes:
            if node not in node_popularity:
                node_popularity[node] = 0
            node_popularity[node] += 1

        node_popularity = \
            {k: v for k, v in sorted(node_popularity.items(),
                                     key=lambda l: l[1], reverse=True)}

        for node in node_popularity:
            node_start = node * loop_bin_size + window_start
            out_file.write(
                f'{chrom.name}\t{node_start}\t{node_start + loop_bin_size}\t{node_popularity[node]}\n')

    with open(f'random_walks/{chrom.sample_name}.{chrom.name}.'
              f'{window_start}-{window_end}.null_start_node.txt',
              'w') as out_file:
        node_popularity = {}
        for node in null_start_nodes:
            if node not in node_popularity:
                node_popularity[node] = 0
            node_popularity[node] += 1

        node_popularity = \
            {k: v for k, v in sorted(node_popularity.items(),
                                     key=lambda l: l[1], reverse=True)}

        for node in node_popularity:
            node_start = node * loop_bin_size + window_start
            out_file.write(
                f'{chrom.name}\t{node_start}\t{node_start + loop_bin_size}\t{node_popularity[node]}\n')

    walk_graph(start_nodes, adj_list_dict, walk_len, window_start,
               loop_bin_size,
               path_list, bin_popularity, False, max(node_weights),
               max_loop_bin_value)

    walk_graph(start_nodes, adj_list_dict, walk_len, window_start,
               loop_bin_size,
               edge_null_walks, edge_null_popularity, True, max(node_weights),
               max_loop_bin_value, total_potential_nodes)

    walk_graph(null_start_nodes, adj_list_dict, walk_len, window_start,
               loop_bin_size, all_null_walks, all_null_popularity, True,
               1 / len(total_potential_nodes),
               max_loop_bin_value, total_potential_nodes)

    bin_popularity = \
        {k: v for k, v in sorted(bin_popularity.items(),
                                 key=lambda l: l[1], reverse=True)}

    output_random_walk_loops(walk_iter, chrom.window_size, loop_bin_size,
                             'normal', bin_popularity)

    output_random_walk_loops(walk_iter, chrom.window_size, loop_bin_size,
                             'edge_null', edge_null_popularity)

    output_random_walk_loops(walk_iter, chrom.window_size, loop_bin_size,
                             'all_null', all_null_popularity)

    return path_list, bin_popularity


def combine_random_walks(sample_popularity):
    combined_popularity_dict = {}
    for sample_num, sample in enumerate(sample_popularity):
        combined_popularity_dict[sample] = {}
        for chrom_name in sample:
            chrom_popularity = sample_popularity[sample][chrom_name]
            combined_chrom_popularity = combined_popularity_dict[sample][
                chrom_name] = {}
            for window_start, window_popularity in chrom_popularity.items():
                for pair in window_popularity:
                    combined_chrom_popularity[pair] = window_popularity[pair]

                    if window_start == 0:
                        combined_chrom_popularity[pair] += window_popularity[
                            pair]

    return combined_popularity_dict


def output_random_walk_loops(walk_iter, window_size, loop_bin_size,
                             walk_type, popularity_dict):

    if not os.path.isdir('random_walks/bin_loops'):
        os.mkdir('random_walks/bin_loops')

    for sample_name in popularity_dict:
        with open(f'random_walks/bin_loops/{sample_name}.'
                  f'walk_iter={walk_iter}.window_size={window_size}.'
                  f'loop_bin_size={loop_bin_size}.{walk_type}.forward.loops',
                  'w') as forward_file, \
                open(f'random_walks/bin_loops/{sample_name}.'
                     f'walk_iter={walk_iter}.window_size={window_size}.'
                     f'loop_bin_size={loop_bin_size}.{walk_type}.back.loops',
                     'w') as back_file:

            for chrom_name in popularity_dict[sample_name]:
                chrom_dict = popularity_dict[sample_name][chrom_name]

                for key in chrom_dict:
                    anchors = key.split('-')
                    start = int(anchors[0])
                    end = int(anchors[1])

                    if end > start:
                        forward_file.write(
                            f'{chrom_name}\t{start}\t{start + loop_bin_size}\t'
                            f'{chrom_name}\t{end}\t{end + loop_bin_size}\t'
                            f'{chrom_dict[key]}\n')

                    elif start > end:
                        back_file.write(
                            f'{chrom_name}\t{end}\t{end + loop_bin_size}\t'
                            f'{chrom_name}\t{start}\t{start + loop_bin_size}\t'
                            f'{chrom_dict[key]}\n')

                    else:
                        log.warning(f'Detected self loop: {key}. Maybe bug in '
                                    f'code. Should have removed when creating '
                                    f'adj list')


def compare_random_walks(walk_iter, walk_len, sample_popularity=None,
                         popularity_folder=None,
                         sort_by=None, chrom_name='chr1'):
    if sort_by and sort_by not in sample_popularity:
        raise RuntimeError(f'{sort_by} is not a valid sample name in the first '
                           f'parameter')

    if not sample_popularity and not popularity_folder:
        raise RuntimeError('Not given any popularities to compare')

    if sample_popularity and popularity_folder:
        raise RuntimeError('Not sure which popularity to compare')

    sample_list = list(sample_popularity.keys())
    comparison_list = list(itertools.combinations(sample_list, 2))
    sample_list_order = []

    if sample_popularity:
        popularity_dict = {}
        for sample_num, sample in enumerate(sample_popularity):
            sample_list_order.append(sample)
            chrom_popularity = sample_popularity[sample][chrom_name]
            for window_start, window_popularity in chrom_popularity.items():
                if window_start not in popularity_dict:
                    popularity_dict[window_start] = {}
                window_popularity_list = popularity_dict[window_start]

                for pair in window_popularity:
                    if pair not in window_popularity_list:
                        window_popularity_list[pair] = \
                            [0 for _ in range(sample_num)] + \
                            [window_popularity[pair]]
                    else:
                        window_popularity_list[pair].append(
                            window_popularity[pair])

                for pair in window_popularity_list:
                    if len(window_popularity_list[pair]) <= sample_num:
                        window_popularity_list[pair].append(0)

        for window_start, window_popularity in popularity_dict.items():
            with open(f'random_walks/total_popularity_comparison.'
                      f'window_start={window_start}.walk_iter={walk_iter}.'
                      f'walk_len={walk_len}.txt', 'w') as out_file:

                popularity = prettytable.PrettyTable()
                popularity.field_names = ['Pair'] + list(sample_popularity) + [
                    f'{x[0]}/{x[1]}' for x in comparison_list
                ]
                for pair, popularity_list in window_popularity.items():
                    for comparison in comparison_list:
                        sample1 = comparison[0]
                        sample2 = comparison[1]
                        sample1_index = sample_list_order.index(sample1)
                        sample2_index = sample_list_order.index(sample2)
                        sample1_popularity = popularity_list[sample1_index]
                        sample2_popularity = popularity_list[sample2_index]

                        if sample2_popularity == 0:
                            sample2_popularity = 1

                        popularity_list.append((sample1_popularity + 100) /
                                               (sample2_popularity + 100))

                    popularity.add_row([pair] + popularity_list)

                popularity.sortby = popularity.field_names[
                    len(sample_popularity) + 1]
                popularity.reversesort = True

                out_file.write(popularity.get_string())

    elif popularity_folder:
        # TODO: Read popularity from folder
        log.error(f'Reading popularity from a folder is not yet implemented')
