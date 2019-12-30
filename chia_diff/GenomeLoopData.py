import numpy as np
import os
import logging

from .ChromLoopData import ChromLoopData

VERSION = 17
log = logging.getLogger()
log_bin = logging.getLogger('bin')

# Missing in many miseq peak files
CHROMS_TO_IGNORE = ['chrY', 'chrM']


class GenomeLoopData:
    """
    A class used to represent a sample

    Attributes
    ----------
    species_name : str
        Name of the sample's species (hg38, mm10, ...)
    sample_name : str
        Name of the sample (LHH0061, LHH0061_0061H, ...)
    chrom_dict : dict(str, ChromLoopData)
        Key: Name of chromosome
        Value: ChromLoopData object
    """

    def __init__(self, chrom_size_file_path, loop_file_path, bedgraph,
                 chroms_to_load=None, min_loop_value=0):
        """
        Initializes all chromosomes and adds loops to them from given file.

        Parameters
        ----------
        chrom_size_file_path : str
            File containing the base pair size of each chromosome to use
        loop_file_path : str
            File containing loops in format:
            chrom1  start1   end1 chrom2  start2   end2 pet_count
        bedgraph : BedGraph
            The bedgraph file for this sample (from pyBedGraph)
        chroms_to_load : list, optional
             List of names of chromosome to load (default is None)
        min_loop_value : int, optional
            Minimum loop value (PET count) to include (default is 0)
        """

        # log.debug(locals())

        self.species_name = os.path.basename(chrom_size_file_path).split('.')[0]
        self.sample_name = os.path.basename(loop_file_path).split('.')[0]

        self.window_size = None

        self.total_samples = 0

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
                    np.sum(self.chrom_dict[chrom_name].pet_count_list)
            else:
                to_remove.append(chrom_name)

        # Chromosomes with no loops or other random problems
        for chrom_name in to_remove:
            del self.chrom_dict[chrom_name]

    def preprocess(self, window_size):
        """
        Preprocess all the chromosomes in this object.

        Removes all problematic chromosomes (not enough loops, etc...).

        Parameters
        ----------
        window_size : int
            The window size to separate the counting of loop count
        """

        self.window_size = window_size

        to_remove = []
        for name, chrom_data in self.chrom_dict.items():
            if not chrom_data.preprocess(window_size=window_size):
                to_remove.append(name)

        # Remove problematic chromosomes
        for name in to_remove:
            log.warning(f'Removing {name} because there are problems with it')
            del self.chrom_dict[name]
