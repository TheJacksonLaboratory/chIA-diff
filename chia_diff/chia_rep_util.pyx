from libc.math cimport fabs
import numpy as np

def c_emd(double[:] p, double[:] q, unsigned int size):
    """
    Calculates Earth Mover's Distance in Cython

    Iterates through array, keeping track of the differences

    Parameters
    ----------
    p
        The row or column of sample1
    q
        The matching row or column of sample2
    size
        The size of p,q. They should be the same size.

    Returns
    -------
    float
        The EMD
    """

    cdef Py_ssize_t i

    cdef double to_move = 0, dist = 0
    for i in range(size):
        to_move += p[i] - q[i]
        dist += fabs(to_move)

    return dist

def match_graphs(double[:, :] p, double[:, :] q, double[:, :] p1,
                 double[:, :] q1, unsigned int size):
    cdef Py_ssize_t i, j

    for i in range(size):
        for j in range(size):
            if p[i][j] == 0:
                q1[i][j] = 0

            if q[i][j] == 0:
                p1[i][j] = 0

def get_loops(unsigned int window_start, unsigned int window_end,
              unsigned char[:] removed_area, int[:] start_list,
              int[:] end_list, double[:] value_list):
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

def complete_graph(double[:, :] graph, int graph_size):
    cdef size_t i, j

    for i in range(graph_size):
        for j in range(i):
            graph[i][j] = graph[j][i]
