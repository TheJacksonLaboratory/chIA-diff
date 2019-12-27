import os
from pyBedGraph import BedGraph
import numpy as np

PEAK_DATA_DIR = '/media/hirwo/extra/jax/data/chia_pet/peaks'
LOOP_DATA_DIR = '/media/hirwo/extra/jax/data/chia_pet/loops'
BEDGRAPH_DIR = '/media/hirwo/extra/jax/data/chia_pet/bedgraphs'

MY_PEAK_DATA_DIR = '/media/hirwo/extra/jax/data/chia_pet/my_full_peaks'
MY_LOOP_DATA_DIR = '/media/hirwo/extra/jax/data/chia_pet/my_loops'


HUMAN_CHROM_SIZE_FILE = \
    '/media/hirwo/extra/jax/data/chia_pet/chrom_sizes/hg38.chrom.sizes'
MOUSE_CHROM_SIZE_FILE = \
    '/media/hirwo/extra/jax/data/chia_pet/chrom_sizes/mm10.chrom.sizes'

hg_sample_names = []
mm_sample_names = []


def get_file_path(sample_name, data_dir):
    for file_name in os.listdir(data_dir):
        if sample_name in file_name:
            return os.path.join(data_dir, file_name)


def read_peak_file(peak_file_path, is_narrow):
    peak_dict = {}
    max_value = -1
    with open(peak_file_path) as peak_file:
        for line in peak_file:
            line = line.split()
            chrom_name = line[0]

            try:
                peak_start = int(line[1])
                peak_end = int(line[2])
                if is_narrow:  # peak value is q-value
                    peak_value = float(line[8])
                else:
                    peak_value = float(line[6])
            except ValueError:
                print(line)
                continue

            if chrom_name not in peak_dict:
                peak_dict[chrom_name] = {
                    'start_list': [],
                    'end_list': [],
                    'value_list': []
                }
            if peak_start > max_value:
                max_value = peak_start
            if peak_end > max_value:
                max_value = peak_end

            peak_dict[chrom_name]['start_list'].append(peak_start)
            peak_dict[chrom_name]['end_list'].append(peak_end)
            peak_dict[chrom_name]['value_list'].append(peak_value)

    return peak_dict


def read_bedgraph(sample_name, chrom_file):
    bedgraph_file_path = get_file_path(sample_name, BEDGRAPH_DIR)

    if bedgraph_file_path is None:
        print(f"No bedgraph file for {sample_name}")
        return None

    print(f"Reading in {bedgraph_file_path}")
    # return BedGraph(chrom_file, bedgraph_file_path, ignore_missing_bp=False, chrom_wanted='chr1')
    return BedGraph(chrom_file, bedgraph_file_path, ignore_missing_bp=False)


def main():
    with open('/media/hirwo/extra/jax/data/chia_pet/hg_libraries.txt') as f:
        for line in f:
            hg_sample_names.append(line.strip())

    with open('/media/hirwo/extra/jax/data/chia_pet/mm_libraries.txt') as f:
        for line in f:
            mm_sample_names.append(line.strip())

    for peak_file_name in os.listdir(PEAK_DATA_DIR):
        if 'peak' not in peak_file_name.lower():
            continue

        is_narrow = False
        if 'narrowpeak' in peak_file_name.lower():
            is_narrow = True

        sample_name = peak_file_name.split('.')[0]

        if os.path.isfile(f'{MY_PEAK_DATA_DIR}/{sample_name}.mypeak'):
            continue

        loop_file_path = LOOP_DATA_DIR + f'/{sample_name}.e500.clusters.cis.BE3'

        chrom_file = None
        if sample_name in hg_sample_names:
            chrom_file = HUMAN_CHROM_SIZE_FILE
        if sample_name in mm_sample_names:
            if chrom_file:
                print(sample_name)
                print("ERROR")
            chrom_file = MOUSE_CHROM_SIZE_FILE

        if not chrom_file:
            print(f"{sample_name} is not a mouse or a human. Skipping")
            continue

        print(f"Finding peaks for {sample_name}")
        peak_file_path = os.path.join(PEAK_DATA_DIR, peak_file_name)
        peak_dict = read_peak_file(peak_file_path, is_narrow)

        b = read_bedgraph(sample_name, chrom_file)
        if b is None:
            continue

        temp_array = np.zeros(1000000, dtype=np.int8)
        for chrom_name, peaks in peak_dict.items():
            '''if chrom_name != 'chr1':
                peaks['max_values'] = temp_array
                peaks['mean_values'] = temp_array
                continue'''
            b.load_chrom_data(chrom_name)
            try:
                peaks['max_values'] = b.stats(start_list=peaks['start_list'],
                                              end_list=peaks['end_list'],
                                              stat='max', chrom_name=chrom_name)

                peaks['mean_values'] = b.stats(start_list=peaks['start_list'],
                                               end_list=peaks['end_list'],
                                               stat='mean', chrom_name=chrom_name)
            except IndexError:
                max_value = -1
                print(chrom_name)
                for i in range(len(peaks['start_list'])):
                    assert peaks['start_list'][i] < peaks['end_list'][i]

                    if peaks['end_list'][i] > max_value:
                        max_value = peaks['end_list'][i]
                print(max_value)
            b.free_chrom_data(chrom_name)

        with open(f'{MY_PEAK_DATA_DIR}/{sample_name}.mypeak', 'w+') as out_file:
            for chrom_name, peaks in peak_dict.items():
                for i in range(len(peaks['end_list'])):
                    out_file.write(f'{chrom_name}\t{peaks["start_list"][i]}\t'
                                   f'{peaks["end_list"][i]}\t'
                                   f'{peaks["value_list"][i]}\t'
                                   f'{peaks["max_values"][i]}\t'
                                   f'{peaks["mean_values"][i]}\n')


if __name__ == "__main__":
    main()