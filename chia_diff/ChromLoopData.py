from .chia_diff import *

log = logging.getLogger()

MIN_NUMB_LOOPS = 5
MAX_LOOP_LEN = 1000000  # 1mb
MIN_LOOP_LEN = 8000  # 8kb
MAX_AVG_ANCHOR_LEN = 5000

VERSION = 61

MAX_USHRT = 65535
MIN_RATIO_INCREASE = 1.1

MIN_PEAK_VALUE = 70


class ChromLoopData:
    """
    A class used to hold loop information in a sample's chromosome

    Attributes
    ----------
    name : str
        Name of chromosome
    size : int
        Size of chromosome
    sample_name : str
        Name of the sample (LHH0061, LHH0061_0061H, ...)
    """

    def __init__(self, chrom_name, chrom_size, sample_name,
                 coverage_bin_size=COVERAGE_BIN_SIZE):
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
        self.start_anchor_weights = None
        self.end_anchor_weights = None
        self.start_list = None  # Later initialized with bedgraph
        self.end_list = None
        self.pet_count_list = []
        self.numb_loops = 0

        self.removed_intervals = [[], []]  # Start/end anchors are on same peak
        self.start_list_peak_values = None
        self.end_list_peak_values = None

        self.coverage_bins = None
        self.coverage_bin_size = coverage_bin_size

        self.window_loop_list = []
        self.window_total_loop_count = []
        self.window_size = None
        self.numb_windows = None
        self.window_coverage_list = []  # Holds a coverage array for each window

        self.max_loop_value = 0

    def add_loop(self, loop_start1, loop_end1, loop_start2, loop_end2,
                 loop_value):
        avg_anchor_len = (
            (loop_end1 - loop_start1) + (loop_end2 - loop_start2)
        ) / 2

        # Loop span: end of (start anchor) -> start of (end anchor)
        if loop_start2 - loop_end1 < MIN_LOOP_LEN and \
                avg_anchor_len > MAX_AVG_ANCHOR_LEN:
            return

        self.start_anchor_list[0].append(loop_start1)
        self.start_anchor_list[1].append(loop_end1)
        self.end_anchor_list[0].append(loop_start2)
        self.end_anchor_list[1].append(loop_end2)

        self.pet_count_list.append(loop_value)

        self.numb_loops += 1

    def finish_init(self, bedgraph):
        """
        Finishes the construction of this chromosome. Converts lists to numpy
        arrays to save memory.

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

        # if not bedgraph.has_chrom(self.name):
        if self.name not in bedgraph.chromosome_map:
            log.warning(f"{self.name} was not found in corresponding bedgraph: "
                        f"{bedgraph.name}")
            return False

        self.pet_count_list = np.asarray(self.pet_count_list, dtype=np.int32)
        self.start_anchor_list = np.asarray(self.start_anchor_list,
                                            dtype=np.int32)
        self.end_anchor_list = np.asarray(self.end_anchor_list, dtype=np.int32)

        # Later populated in preprocessing
        self.start_anchor_weights = np.zeros(self.numb_loops)
        self.end_anchor_weights = np.zeros(self.numb_loops)

        log.debug(f"Max PET count: {np.max(self.pet_count_list)}")

        self.find_loop_anchor_points(bedgraph)
        return True

    def find_loop_anchor_points(self, bedgraph):
        """
        Finds the max coverage for each small non-overlapping bins over the
        entire chromosome.

        Also finds the exact loop start/end point within anchors by finding the
        index of max value within anchors.

        Also finds the max value inside every anchor. (currently unused)

        Parameters
        ----------
        bedgraph : BedGraph
            Used to find the coverage values for anchor of each loop
        """

        log.info(f'Finding anchor points for {self.sample_name}\'s {self.name}'
                 f' from {bedgraph.name}')

        bedgraph.load_chrom_data(self.name)

        coverage_bin_starts = np.arange(0, self.size, self.coverage_bin_size,
                                        dtype=np.int32)
        coverage_bin_ends = coverage_bin_starts + self.coverage_bin_size
        if coverage_bin_ends[-1] > self.size:
            coverage_bin_ends[-1] = self.size
        self.coverage_bins = bedgraph.stats(start_list=coverage_bin_starts,
                                            end_list=coverage_bin_ends,
                                            chrom_name=self.name, stat='max')

        # Somewhat arbitrary way of filtering out bedgraph files
        # Based on median/max of bedgraph?
        # TODO: Make this better/applicable to all samples
        min_peak_value = 0
        if 'LHM0008' in self.sample_name:
            min_peak_value = 57
        elif 'LHM0011' in self.sample_name:
            min_peak_value = 47

        for i in range(len(self.coverage_bins)):
            if self.coverage_bins[i] <= min_peak_value:
                self.coverage_bins[i] = 0

        output_coverage_bins(self.sample_name, self.name, self.coverage_bins,
                             self.coverage_bin_size)

        # Get index of peak in each anchor interval
        self.start_list = bedgraph.stats(start_list=self.start_anchor_list[0],
                                         end_list=self.start_anchor_list[1],
                                         chrom_name=self.name, stat='max_index')
        self.end_list = bedgraph.stats(start_list=self.end_anchor_list[0],
                                       end_list=self.end_anchor_list[1],
                                       chrom_name=self.name, stat='max_index')

        # Get peak value for every anchor interval (currently unused)
        start_list_peaks = bedgraph.stats(start_list=self.start_anchor_list[0],
                                          end_list=self.start_anchor_list[1],
                                          chrom_name=self.name, stat='max')
        end_list_peaks = bedgraph.stats(start_list=self.end_anchor_list[0],
                                        end_list=self.end_anchor_list[1],
                                        chrom_name=self.name, stat='max')
        self.start_list_peak_values = start_list_peaks
        self.end_list_peak_values = end_list_peaks
        bedgraph.free_chrom_data(self.name)

        # start_list_peaks = start_list_peaks / start_list_peaks.sum()
        # end_list_peaks = end_list_peaks / end_list_peaks.sum()

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
                # Currently unused since smaller loop lengths are filtered
                # earlier
                self.pet_count_list[i] = 0

                # Removed interval goes from
                # (start of start anchor, end of end anchor)
                self.removed_intervals[0].append(self.start_anchor_list[0][i])
                self.removed_intervals[1].append(self.end_anchor_list[1][i])
                continue

            # Weigh each loop based on its corresponding bedgraph peak
            # peak_value = max(start_list_peaks[i], end_list_peaks[i])
            # peak_value = start_list_peaks[i] * end_list_peaks[i]
            # self.pet_count_list[i] *= peak_value

            # Remove loops over a given threshold
            # loop_length = int(loop_end) - int(loop_start)
            # if loop_length > MAX_LOOP_LEN:
            #     self.pet_count_list[i] = 0

        self.max_loop_value = np.max(self.pet_count_list)

        log.debug(f"Max loop weighted value: {self.max_loop_value}")

    def preprocess(self, window_size=None):
        """
        Finds the loops in each window and adds weights to their anchors

        Parameters
        ----------
        window_size : int, optional
            The size of the sliding window
            (Default is the size of the chromosome)
        """

        if not window_size:
            window_size = self.size

        self.window_size = window_size
        self.numb_windows = ceil(self.size / window_size)

        # if window_size % PEAK_BIN_LEN != 0:
        #     log.error(f'Window size: {window_size} must be a multiple of '
        #               f'PEAK_BIN_LEN: {PEAK_BIN_LEN}')
        #     return False

        if window_size < self.coverage_bin_size:
            log.error(f'Window size: {window_size} must be greater than '
                      f'Coverage bin size: {self.coverage_bin_size}')
            return False

        self.find_window_loops()
        self.weight_loops()

        return True

    def find_window_loops(self):
        """
        Finds the indexes of the loops within each window and stores them in
        self.window_loop_list. Also stores the total PET count in each window in
        self.window_total_loop_count.
        """

        self.window_loop_list.clear()
        self.window_total_loop_count.clear()

        numb_windows = ceil(self.size / self.window_size)
        for i in range(numb_windows):
            window_start = i * self.window_size
            window_end = window_start + self.window_size
            loop_indexes = get_loops(window_start, window_end,
                                     np.zeros(self.size, dtype=np.uint8),
                                     self.start_list, self.end_list,
                                     self.pet_count_list)

            total_value = 0
            for loop_index in loop_indexes:
                total_value += self.pet_count_list[loop_index]

            self.window_loop_list.append(loop_indexes)
            self.window_total_loop_count.append(total_value)

    def weight_loops(self, output_dir='weighted_loops'):
        """
        Weights each loop based on coverage within its anchors.
        Each window has its own normalized list of coverage that is used for
        weighting the loops.

        Parameters
        ----------
        output_dir : str, optional
            Directory to output weighted loops (Default is None)
        """

        self.window_coverage_list.clear()

        out_file = None
        if output_dir:
            out_file = open(f'{output_dir}/{self.sample_name}.txt', 'w')

        for i in range(self.numb_windows):
            log.debug(f'Weighting loops in Window {i}')

            window_start = i * self.window_size
            window_end = window_start + self.window_size

            start = int(window_start / self.coverage_bin_size)
            end = ceil(window_end / self.coverage_bin_size)

            numb_loops = self.window_loop_list[i].size

            if numb_loops == 0:
                log.info('No loops in this window')
                continue

            # Get the coverages only from this window
            window_coverage_list = np.array(self.coverage_bins[start:end])
            # log.debug(f'Unweighted peak values: {peak_list}')

            # Make all loops are initialized to have 0 weight
            if window_coverage_list.sum() == 0:
                continue

            # Normalize the list specific to this window
            window_coverage_list /= window_coverage_list.sum()
            # log.debug(f'Weighted peak values: {peak_list}')

            self.window_coverage_list.append(window_coverage_list)

            # Get a continuous list of the anchors in this window
            start_anchors = (np.empty(numb_loops, dtype=np.int32),
                             np.empty(numb_loops, dtype=np.int32))
            end_anchors = (np.empty(numb_loops, dtype=np.int32),
                           np.empty(numb_loops, dtype=np.int32))
            for j, loop_index in enumerate(self.window_loop_list[i]):
                start_anchors[0][j] = self.start_anchor_list[0][loop_index]
                start_anchors[1][j] = self.start_anchor_list[1][loop_index]
                end_anchors[0][j] = self.end_anchor_list[0][loop_index]
                end_anchors[1][j] = self.end_anchor_list[1][loop_index]

                # Part of the anchor may be outside the window even though the
                # peak index of the anchor is inside
                if start_anchors[0][j] < window_start:
                    start_anchors[0][j] = window_start
                if end_anchors[1][j] > window_end:
                    end_anchors[1][j] = window_end

            # log.debug(f'Start anchors: {start_anchors}')
            # log.debug(f'End anchors: {end_anchors}')

            start_weights = get_total_bin_value(
                window_coverage_list, COVERAGE_BIN_SIZE,
                start_anchors[0] - window_start, start_anchors[1] - window_start)
            # log.debug(f'Start anchor weights: {start_weights}')

            end_weights = get_total_bin_value(
                window_coverage_list, COVERAGE_BIN_SIZE,
                end_anchors[0] - window_start, end_anchors[1] - window_start)
            # log.debug(f'End anchor weights: {end_weights}')

            # Populate the original array
            for j, loop_index in enumerate(self.window_loop_list[i]):
                self.start_anchor_weights[loop_index] = start_weights[j]
                self.end_anchor_weights[loop_index] = end_weights[j]

        if out_file:
            out_str = ''
            for j in range(self.numb_loops):
                out_str += (f'{self.name}\t'
                            f'{self.start_anchor_list[0][j]}\t'
                            f'{self.start_anchor_list[1][j]}\t'
                            f'{self.start_list[j]}\t'
                            f'{self.name}\t'
                            f'{self.end_anchor_list[0][j]}\t'
                            f'{self.end_anchor_list[1][j]}\t'
                            f'{self.end_list[j]}\t'
                            f'{self.pet_count_list[j]}\t'
                            f'{round(self.start_anchor_weights[j], 5)}\t'
                            f'{round(self.end_anchor_weights[j], 5)}\n')
            out_file.write(out_str)

        if out_file:
            out_file.close()

    # def create_diff_graph(self, loops, bin_size, window_size, peak_arr):
    #     graph_len = ceil(window_size / bin_size)
    #
    #     # 2D array of arrays
    #     # (none, single, both) peak support
    #     graph = [[([], [], []) for _ in range(graph_len)] for _ in
    #              range(graph_len)]
    #
    #     for loop_index in loops:
    #
    #         value = self.pet_count_list[loop_index]
    #         start = self.start_list[loop_index]
    #         end = self.end_list[loop_index]
    #
    #         window_start = start % window_size
    #         window_end = end % window_size
    #
    #         # loop_len[int((end - start) / self.bin_size)] += 1
    #
    #         bin_start = int(window_start / bin_size)
    #         bin_end = int(window_end / bin_size)
    #
    #         peak_supp = 0
    #         if peak_arr[start]:
    #             peak_supp += 1
    #         if peak_arr[end]:
    #             peak_supp += 1
    #
    #         graph[bin_start][bin_end][peak_supp].append([start, end, value])
    #
    #         # if bin_end < bin_start:
    #         #     log.error(
    #         #         f'{orig_start}\t{orig_end}\t{start}\t{end}\t{bin_start}\t{bin_end}')
    #
    #         # Get the other side of the graph as well for emd calculation
    #         # graph[bin_end][bin_start] += value
    #
    #         # Avoid double counting the middle
    #         # if bin_end != bin_start:
    #         #    graph[bin_end][bin_start] += value
    #
    #         # Also get areas surrounding this loop
    #         # May not be needed with emd calculation
    #         # Helps with finding jensen-shannon
    #         # for j in range(bin_start - 1, bin_start + 2):
    #         #     if j < 0 or j == graph_len:
    #         #         continue
    #         #     for k in range(bin_end - 1, bin_end + 2):
    #         #         if k < 0 or k == graph_len:
    #         #             continue
    #         #         graph[j][k] += value
    #         # graph[k][j] += value
    #         # if j != k:
    #         #    graph[k][j] += value
    #
    #         # if to_debug:
    #         #     log.debug(
    #         #         f'{self.sample_name}\t{orig_start}\t{orig_end}\t{value}')
    #
    #     for i in graph_len:
    #         for j in graph_len:
    #             for k in 3:
    #                 graph[i][j][k].sort(key=lambda l: l[0])
    #
    #                 loop_bin = graph[i][j][k]
    #                 merged_list = []
    #
    #                 # Merge all loops with the same peak
    #                 # Loops anchors already set to peak in preprocessing
    #                 for key, group in itertools.groupby(loop_bin,
    #                                                     lambda l: l[0]):
    #                     total_sum = 0
    #                     loop_end = None
    #                     for arr in group:
    #                         total_sum += arr[2]
    #                         if not loop_end:
    #                             loop_end = arr[1]
    #                         if loop_end != arr[1]:
    #                             log.error("")
    #
    #                     merged_list.append((key, group[0], sum))
    #                     curr_start = loop_bin[loop_index][0]
    #                     prev_start = loop_bin[loop_index - 1][0]
    #
    #                     if curr_start == prev_start:
    #                         continue
    #
    #     # log.info(f"Number of loops in {self.sample_name} graph: {num_loops_used}")
    #
    #     # plt.plot([x for x in range(len(loop_len))], [np.log(x) for x in loop_len])
    #     # plt.show()
    #
    #     # log.info(f'Max value in graph: {np.max(graph)}')
    #     # return graph, total_PET_count / self.total_loop_value
    #     return graph

    # Weight each
    # def create_peak_graph(self, merged_list):
    #     graph_len = len(merged_list)
    #     graph = np.zeros((graph_len, graph_len), dtype=np.float64)
    #
    #     peak_array = np.full(self.size, -1, dtype=np.int16)
    #     for i, peak in enumerate(merged_list):
    #         start = peak[PEAK_START_INDEX]
    #         end = peak[PEAK_END_INDEX]
    #         peak_array[start:end] = i
    #
    #     log.info(f'Numb of loops: {self.numb_loops}')
    #     for loop_index in range(self.numb_loops):
    #         value = self.pet_count_list[loop_index]
    #         start = self.start_list[loop_index]
    #         end = self.end_list[loop_index]
    #
    #         start_index = peak_array[start]
    #         end_index = peak_array[end]
    #
    #         if start_index == -1 or end_index == -1:
    #             log.critical(f"Loop: ({start}, {end}, {value}) "
    #                          f"does not have a corresponding peak")
    #             continue
    #
    #         graph[start_index][end_index] += value
    #
    #     return graph / graph.sum()

#  END
