import os
import numpy as np
import sys
import logging
from logging.config import fileConfig
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
from scipy.optimize import curve_fit

sys.path.append('..')

from chia_diff.util import read_data

LOOP_DATA_DIR = '/media/hirow/extra/jax/data/to_use'
BIGWIG_DATA_DIR = '/media/hirow/extra/jax/data/to_use'
CHROM_DATA_DIR = '/media/hirow/extra/jax/data/chrom_sizes'

fileConfig('../test/chia_diff.conf')
log = logging.getLogger()
log.setLevel(logging.DEBUG)
mpl.rcParams['savefig.dpi'] = 400

VERSION = 3

PET_CATEGORIES = ['PET Count 2-5', 'PET Count 6-10', 'PET Count > 10']


def coverage_bin_density(sample_name, coverage_bins):
    # if 'LHH0085' in sample_name:
    #     return
    print(sample_name)
    # coverage_bins = [x for x in coverage_bins if x != 0]
    coverage_bins = [x for x in coverage_bins if x != 0]
    less_than_10 = 0
    for coverage in coverage_bins:
        if coverage < 10:
            less_than_10 += 1

    print(less_than_10 / len(coverage_bins))

    (n, bins, patches) = plt.hist(np.log10(coverage_bins), density=1, bins=25)
    print(bins)
    print()
    plt.ylabel('Frequency')
    plt.xlabel('Max in bin (log10)')
    plt.title(f'{sample_name} Coverage Bin Density')
    plt.savefig(f'{sample_name}_coverage_density.png')
    plt.close()


def anchor_len_density(sample_name, anchor_lengths):
    THRESHOLD = 3.5

    numb_bins = 40
    anchor_len_density = [0 for _ in range(numb_bins)]
    above_threshold = 0
    max_anchor_len = np.max(anchor_lengths)
    assert max_anchor_len * 2 < 10
    for anchor_len in anchor_lengths:

        if anchor_len >= THRESHOLD:
            above_threshold += 1

    bin_numbers = [x for x in range(numb_bins)]
    print(anchor_len_density)
    while len(anchor_len_density) > 0:
        if anchor_len_density[0] == 0:
            anchor_len_density.pop(0)
            bin_numbers.pop(0)
            continue
        elif anchor_len_density[-1] == 0:
            anchor_len_density.pop()
            bin_numbers.pop()
            continue
        else:
            break
    print(anchor_len_density)
    numb_bins = len(bin_numbers)

    # plt.hist(np.array(bin_numbers), len(bin_numbers),
    #          weights=np.array(anchor_len_density), density=True)
    plt.hist(anchor_lengths, density=1, bins=10)
    plt.ylabel('Frequency')
    plt.xlabel('Anchor Length')
    plt.text(0.75, 0.75,
             f'Max Anchor Length: {max_anchor_len}\nPercentage of Anchors > '
             f'{THRESHOLD}: {above_threshold / len(anchor_lengths)}',
             horizontalalignment='center',
             verticalalignment='center', transform=plt.gca().transAxes)
    plt.title(f'{sample_name} Loop Anchor Length Frequency')
    plt.savefig(f'{sample_name}_anchor_len_hist.png')
    plt.close()


def anchor_len_pet_count(sample_name, anchor_len, pet_count):
    def func(x, a, b):
        return a * np.exp(b * x)

    initial_guess = [0.00001, 3]
    popt, pcov = curve_fit(func, anchor_len, pet_count, initial_guess)
    sample_anchor_len = np.linspace(min(anchor_len), max(anchor_len))
    fitted_data = [func(x, *popt) for x in sample_anchor_len]

    plt.scatter(anchor_len, pet_count, label='Loop Data Point')
    plt.plot(sample_anchor_len, fitted_data, linestyle='-', color='#900000',
             label="{0:0.2g} * e^({1:0.2g} * x))".format(*popt))
    plt.ylabel('PET Count')
    plt.xlabel('Average Anchor Length')
    plt.title(f'{sample_name} Average Anchor Lengths vs. PET Count')
    plt.legend(loc=0, fontsize=12)
    plt.savefig(f'graphs/{sample_name}_anchor_len_PET_count.png')
    plt.close()


def pet_count_hist(sample_name, pet_count):
    numb_bins = 9
    pet_freq = np.zeros(numb_bins + 3, dtype=np.int32)
    max_PET = -1
    ge_than_3 = 0
    for pet in pet_count:
        if pet > 10:
            pet_freq[-1] += 1
        else:
            pet_freq[pet] += 1

        if pet >= 3:
            ge_than_3 += 1

        if pet > max_PET:
            max_PET = pet
    plt.hist(np.arange(numb_bins), numb_bins, weights=pet_freq[3:],
             density=True)
    plt.ylabel('Frequency')
    plt.xlabel('PET Count')
    plt.xticks(np.arange(1 / 3, numb_bins * 0.9 + 1 / 3, step=0.9),
               [str(x) for x in range(3, 11)] + ['>10'])
    plt.text(0.75, 0.75,
             f'PET count=1: {pet_freq[1]}\nPET count=2: {pet_freq[2]}\n'
             f'Max PET count: {max_PET}\nTotal # Loops >= 3: {ge_than_3}',
             horizontalalignment='center',
             verticalalignment='center', transform=plt.gca().transAxes)
    plt.title(f'{sample_name} PET Count Frequency')
    plt.savefig(f'graphs/{sample_name}_PET_count_hist.png')
    plt.close()


def loop_span_pet_cat(sample_name, loop_spans, pet_count):
    loop_cat = [[] for _ in range(12)]
    assert loop_spans.size == pet_count.size

    for i in range(loop_spans.size):
        pet = pet_count[i]
        if pet > 11:
            pet = 11
        loop_cat[pet].append(loop_spans[i])

    categories = [[2, 3, 4, 5], [6, 7, 8, 9, 10], [11]]

    f, plots = plt.subplots(1, len(categories), sharey=True)
    f.suptitle(f'{sample_name} Log10(loop span) by PET Category')
    f.text(0.5, 0.04, 'Log10(Loop span)', ha='center')
    f.text(0.04, 0.5, 'Frequency', va='center', rotation='vertical')
    for i, category in enumerate(categories):
        combined_spans = []
        for index in category:
            combined_spans += loop_cat[index]

        plots[i].hist(combined_spans,
                      bins=int(1 + np.log2(len(combined_spans))),
                      density=True)
        plots[i].set_xlim(3, 9)
        plots[i].set_title(PET_CATEGORIES[i])

    plt.savefig(f'graphs/{sample_name}_loop_span_pet_category')
    plt.close()


def loop_span_all(sample_name, loop_spans):
    plt.title(f'{sample_name} Log10(loop span)')
    plt.xlabel('log10(Loop Span)')
    plt.ylabel('Density')
    plt.hist(loop_spans, bins=int(1 + np.log2(len(loop_spans))), density=True)
    plt.savefig(f'graphs/{sample_name}_loop_span_all')
    plt.close()


def anchor_intensity_loop_span(sample_name, avg_anchor_intensity, loop_spans):
    plt.title(f'{sample_name} Anchor Intensity vs. Loop Span')
    plt.xlabel('log10(Avg. Anchor Intensity)')
    plt.ylabel('log10(Loop Span)')
    plt.scatter(avg_anchor_intensity, loop_spans)
    plt.savefig(f'graphs/{sample_name}_anchor_intensity_loop_span')
    plt.close()


def anchor_intensity_loop_span_pet_cat(sample_name, avg_anchor_intensity,
                                       loop_spans, pet_count):
    loop_cat = [[] for _ in range(12)]
    anchor_cat = [[] for _ in range(12)]
    assert loop_spans.size == pet_count.size

    for i in range(loop_spans.size):
        pet = pet_count[i]
        if pet > 11:
            pet = 11
        loop_cat[pet].append(loop_spans[i])
        anchor_cat[pet].append(avg_anchor_intensity[i])

    categories = [[2, 3, 4, 5], [6, 7, 8, 9, 10], [11]]

    f, plots = plt.subplots(1, len(categories), sharey=True)
    f.suptitle(f'{sample_name} Anchor Intensity vs. Loop Span by PET Category')
    f.text(0.5, 0.04, 'log10(Average Anchor Intensity)', ha='center')
    f.text(0.04, 0.5, 'log10(Loop Span)', va='center', rotation='vertical')
    for i, category in enumerate(categories):
        combined_spans = []
        combined_anchors = []
        for index in category:
            combined_spans += loop_cat[index]
            combined_anchors += anchor_cat[index]

        plots[i].scatter(combined_anchors, combined_spans)
        plots[i].set_xlim(0, 4)
        plots[i].set_title(PET_CATEGORIES[i])

    plt.savefig(f'graphs/{sample_name}_anchor_intensity_loop_span_pet_category')
    plt.close()


def peak_intensity_density_plot(sample_name, peak_values, peak_percentage):
    plt.hist(peak_values, bins=int(1 + np.log2(len(peak_values))), density=True)
    plt.title(f'{sample_name} Peak Intensity Density Plot {peak_percentage}')
    plt.xlabel(f'log10(Peak Intensity (max))')
    plt.ylabel(f'Density')
    plt.savefig(
        f'graphs/{sample_name}_peak_intensity_density_{peak_percentage}.png')
    plt.close()


def peak_dist_plot(sample_name, peak_dists, peak_percentage):
    plt.hist(peak_dists, bins=int(1 + np.log2(len(peak_dists))), density=True)
    plt.title(f'{sample_name} Peak Distance Density Plot {peak_percentage}')
    plt.xlabel(f'log10(Peak-Peak Distance)')
    plt.ylabel(f'Density')
    plt.savefig(
        f'graphs/{sample_name}_peak_distance_density_{peak_percentage}.png')
    plt.close()


def peak_len_plot(sample_name, peak_len, peak_percentage):
    plt.hist(peak_len, bins=int(1 + np.log2(len(peak_len))), density=True)
    plt.title(f'{sample_name} Peak Length Density Plot {peak_percentage}')
    plt.xlabel(f'Peak Length')
    plt.ylabel(f'Density')
    plt.savefig(f'graphs/{sample_name}_peak_len_density_{peak_percentage}.png')
    plt.close()


def main():
    for sample_name in loop_dict:
        chrom_data = loop_dict[sample_name].chrom_dict['chr1']
        log.info(chrom_data.numb_loops)

        # pet_count_hist(sample_name, chrom_data.pet_count_list)

        avg_anchor_lens = []
        anchor_lengths = []
        loop_spans = []
        avg_anchor_intensity = []
        pet_count_list = []
        for i in range(chrom_data.numb_loops):
            loop_span = chrom_data.end_list[i] - chrom_data.start_list[i]

            if loop_span < 1 or chrom_data.pet_count_list[i] == 0:
                continue

            avg_anchor_lens.append(((chrom_data.start_anchor_list[1][i] -
                                     chrom_data.start_anchor_list[0][i]) +
                                    (chrom_data.end_anchor_list[1][i] -
                                     chrom_data.end_anchor_list[0][i])) / 2)

            anchor_lengths.append(chrom_data.start_anchor_list[1][i] -
                                  chrom_data.start_anchor_list[0][i])
            anchor_lengths.append(chrom_data.end_anchor_list[1][i] -
                                  chrom_data.end_anchor_list[0][i])

            # loop_spans.append(loop_span)
            # avg_anchor_intensity.append((chrom_data.start_list_peaks[i] +
            #                            chrom_data.end_list_peaks[i]) / 2)
            pet_count_list.append(chrom_data.pet_count_list[i])

        avg_anchor_lens = np.log10(avg_anchor_lens)
        anchor_lengths = np.log10(anchor_lengths)
        # loop_spans = np.log10(loop_spans)
        # avg_anchor_intensity = np.log10(avg_anchor_intensity)
        pet_count_list = np.array(pet_count_list)

        anchor_len_density(sample_name, anchor_lengths)
        continue

        # pet_count_hist(sample_name, pet_count_list)
        # loop_span_pet_cat(sample_name, loop_spans, pet_count_list)
        # loop_span_all(sample_name, loop_spans)
        # anchor_intensity_loop_span(sample_name, avg_anchor_intensity, loop_spans)
        anchor_len_pet_count(sample_name, avg_anchor_lens, pet_count_list)
        # anchor_intensity_loop_span_pet_cat(sample_name, avg_anchor_intensity,
        # loop_spans, pet_count_list)

    sys.exit()
    for peak_file_name in os.listdir(MY_FULL_PEAK_DATA_DIR):
        if not peak_file_name.endswith('.mypeak'):
            continue

        sample_name = peak_file_name.split('.')[0]

        orig_peaks = []
        with open(MY_FULL_PEAK_DATA_DIR + '/' + peak_file_name) as in_file:
            for line in in_file:
                line = line.split()
                value = float(line[4])
                if value == 0:
                    continue
                name = line[0]
                start = int(line[1])
                end = int(line[2])

                orig_peaks.append([name, start, end, value])

        orig_peaks.sort(key=lambda x: x[3])
        for peak_percentage in [0.005, 0.01, 0.025, 0.05, 1]:
            peaks = orig_peaks[:math.ceil(len(orig_peaks) * peak_percentage)]
            log.info(f"Kept {len(peaks)} out of {len(orig_peaks)}")

            peaks.sort(key=lambda x: x[1])
            peak_len = []
            peak_values = []
            peak_dist = []
            prev_end = -1
            prev_name = None
            for i in range(len(peaks)):
                peak_values.append(np.log10(peaks[i][3]))
                peak_len.append(peaks[i][2] - peaks[i][1])

                if peaks[i][
                    0] != prev_name:  # Start on the second peak of each chrom
                    prev_name = peaks[i][0]
                    prev_end = -1
                    continue

                prev_peak_end = peaks[i - 1][2]
                if prev_peak_end > prev_end:  # one peak might be within another
                    prev_end = prev_peak_end

                curr_peak_start = peaks[i][1]
                if curr_peak_start > prev_end:
                    peak_dist.append(np.log10(curr_peak_start - prev_end))

            peak_dist_plot(sample_name, peak_dist, peak_percentage)
            peak_intensity_density_plot(sample_name, peak_values,
                                        peak_percentage)
            peak_len_plot(sample_name, peak_len, peak_percentage)


if __name__ == '__main__':
    loop_dict = read_data(loop_data_dir=LOOP_DATA_DIR,
                          chrom_size_file=f'{CHROM_DATA_DIR}/hg38.chrom.sizes',
                          bigwig_data_dir=BIGWIG_DATA_DIR,
                          chroms_to_load=['chr1'])
    main()
