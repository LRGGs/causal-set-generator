import cython

cimport numpy as cnp

import numpy as np


cdef list relatives(p, metric, n, r2):
    cdef int n
    cdef float r2, t_max, l1, l2
    cdef cnp.ndarray metric
    cdef cnp.ndarray p

    children = [[] for _ in range(n)]
    parents = [[] for _ in range(n)]
    for i in range(n):
        t_max = 0.5 * (1 + r2 + p[i][0] - p[i][1])
        l1 = (r2 + p[i][0] - p[i][1])
        l2 = (r2 + p[i][0] + p[i][1])
        for j in range(i + 1, n):
            if p[j][0] > t_max:
                break
            if p[j][0] - p[j][1] > l1 and p[j][0] + p[j][1] > l2:
                continue
            dx = p[j] - p[i]
            interval = metric @ dx @ dx
            if -r2 < interval < 0:
                children[i].append(j)
                parents[j].append(i)

    return children

def Cmake_edges(p, metric, n, r2):
    return relatives(p, metric, n, r2)

