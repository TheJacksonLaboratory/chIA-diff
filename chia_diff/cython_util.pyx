import numpy as np

def get_loops(unsigned int window_start, unsigned int window_end,
              unsigned char[:] removed_area, int[:] start_list,
              int[:] end_list, int[:] value_list):
    """
    Gets all loops within the window but not in the removed area.

    Iterates through every filtered loop

    Parameters
    ----------
    window_start
    window_end
    removed_area
        Same size of chromosome
    start_list
        Numpy array of anchor starts for loops
    end_list
        Numpy array of anchor ends for loops
    value_list
        Numpy array of loop values

    Returns
    -------
    Numpy 1D array
        Array of indexes that contains wanted loops from filtered_loops
    """

    assert end_list.size == value_list.size == start_list.size
    cdef Py_ssize_t numb_values = end_list.size, counter = 0, i, start, end
    cdef float value

    loops = np.empty(numb_values, dtype=np.uint32)
    cdef unsigned int[:] loops_view = loops

    for i in range(numb_values):
        value = value_list[i]
        start = start_list[i]
        end = end_list[i]

        if start < window_start or end > window_end or value == 0:
            continue

        if removed_area[start] or removed_area[end]:
            continue

        loops_view[counter] = i
        counter = counter + 1

    return loops[:counter]

def get_total_bin_value(double[:] bin_list, int max_bin_size, int[:] start_list,
                    int[:] end_list):

    assert tuple(start_list.shape) == tuple(end_list.shape)

    cdef size_t i, start, end, bin_end, bin_index
    cdef size_t num_tests = start_list.size
    cdef double total, fraction, value

    result = np.zeros(num_tests, dtype=np.float64)
    cdef double[:] result_view = result

    for i in range(num_tests):
        start = start_list[i]
        end = end_list[i]

        bin_index = <unsigned int>(start / max_bin_size)
        bin_end = <unsigned int>((end - 1) / max_bin_size)

        # special case where interval is within a single bin
        if bin_index == bin_end:
            if bin_list[bin_index] == 0:
                continue
            result_view[i] = bin_list[bin_index] * (end - start) / max_bin_size
            continue

        total = 0

        # first bin
        value = bin_list[bin_index]
        if value > 0:
            fraction = <double>(max_bin_size - start % max_bin_size) / max_bin_size
            total += bin_list[bin_index] * fraction
        bin_index += 1

        # middle bins
        while bin_index < bin_end:
            value = bin_list[bin_index]
            if value > 0:
                total += bin_list[bin_index]

            bin_index += 1

        # last bin
        value = bin_list[bin_index]
        if value > 0:
            fraction = <double>(end % max_bin_size) / max_bin_size
            if fraction == 0:
                fraction = 1
            total += bin_list[bin_index] * fraction

        result_view[i] = total

    return result
