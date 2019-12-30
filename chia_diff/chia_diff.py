import logging
import numpy as np
import os
import itertools
from .cython_util import *
from math import ceil, sqrt
import matplotlib.pyplot as plt

PEAK_START_INDEX = 0
PEAK_END_INDEX = 1
PEAK_LEN_INDEX = 2
PEAK_MAX_VALUE_INDEX = 3
PEAK_MEAN_VALUE_INDEX = 4

log = logging.getLogger()


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
            lower_peak[PEAK_LEN_INDEX] = lower_peak[PEAK_END_INDEX] - \
                                         lower_peak[
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


def get_removed_area(chrom_size, chrom_list):
    """
    Gets removed area from both samples so no loops from that area are compared

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


def create_graph_list(chrom, loop_indexes, bin_size):

    node_dict = {}

    for loop_index in loop_indexes:
        value = chrom.pet_count_list[loop_index]
        start = chrom.start_list[loop_index]
        end = chrom.end_list[loop_index]

        start = start % chrom.window_size
        end = end % chrom.window_size

        bin_start = int(start / bin_size)
        bin_end = int(end / bin_size)

        if bin_start not in node_dict:
            node_dict[bin_start] = {}

        if bin_end not in node_dict[bin_start]:
            node_dict[bin_start][bin_end] = 0

        # TODO: Add weighting here
        node_dict[bin_start][bin_end] += value

    for bin_start, node in node_dict.items():
        bin_ends = list(node.keys())
        loops = list(node.values())

        node_dict[bin_start] = (bin_ends, loops / np.sum(loops))

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

        graph[bin_start][bin_end] += value * chrom.start_anchor_weights[loop_index]
        graph[bin_end][bin_start] += value * chrom.end_anchor_weights[loop_index]

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
                find_chrom_diff_loops(chrom1, chrom2, window_start, window_end,
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
            with open(f'{output_dir}/{output_file_name}.loops', 'w') as out_file:
                for area in diff_areas:
                    out_file.write(f'{area[0]}\t{area[1]}\t{area[2]}\n')


def find_chrom_diff_loops(chrom1, chrom2, window_start, window_end, bin_size,
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

    combined_removed = get_removed_area(chrom1.size, [chrom1, chrom2])

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
    o_graph = create_graph_matrix(chrom2, o_loop_indexes, bin_size, window_start)

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


def random_walk(chrom, bin_size, walk_len=20, walk_iter=5):
    graph_len = ceil(chrom.window_size / bin_size)

    # Around 6GB
    if graph_len * 20 / 1000000000 > 8:
        log.error(f'{chrom.size / bin_size} is too many rows for a graph adj list')
        return None

    combined_removed = get_removed_area(chrom.size, [chrom])

    # Get loops again in case a specific window_start/end was given
    loop_indexes = get_loops(0, chrom.size, combined_removed, chrom.start_list,
                             chrom.end_list, chrom.pet_count_list)

    adj_list = create_graph_list(chrom, loop_indexes, bin_size)

    for i in range(walk_iter):
        # node =
        pass
