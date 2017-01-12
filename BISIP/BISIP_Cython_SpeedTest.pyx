# -*- coding: utf-8 -*-
#!python
#cython: boundscheck=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: wraparound=False
#cython: boundscheck=False

"""
Created on Wed Nov  4 14:05:35 2015

@author:    clafreniereberube@gmail.com
            École Polytechnique de Montréal

Copyright (c) 2015-2016 Charles L. Bérubé

"""

import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free

DTYPE = np.float_
ctypedef np.float_t DTYPE_t

DTYPE2 = np.complex128
ctypedef np.complex128_t DTYPE2_t

cdef DTYPE2_t powc(DTYPE2_t a, DTYPE2_t b):
    return a**b

cdef double complex C_ColeCole(double w_, double m_, double lt_, double c_):
    cdef double complex Z_i
    cdef double complex jay
    jay.real = 0.0
    jay.imag = 1.0
    Z_i = m_*(1 - 1.0/(1 + powc((jay*w_*(10**lt_)),c_)))
    return Z_i


def ColeCole_cyth2(np.ndarray[DTYPE_t, ndim=1] w, DTYPE_t R0, np.ndarray[DTYPE_t, ndim=1] m, np.ndarray[DTYPE_t, ndim=1] lt, np.ndarray[DTYPE_t, ndim=1] c):
    cdef int N = w.shape[0]
    cdef int D = m.shape[0]
    cdef int i, j, k
    cdef int rows = 2;
    cdef int columns = 5;
    Z = <double **> malloc(2 * sizeof(double *));
    for i in range(rows):
        Z[i] = <double *> malloc(N * sizeof(double));
    cdef double complex z_
    for j in range(N):
        z_ = 0
        for i in range(D):
            z_ = z_ + C_ColeCole(w[j], m[i], lt[i], c[i])
        Z[0][j] = R0*(1 - z_).real
        Z[1][j] = R0*(1 - z_).imag
    cdef double [:,::1] outData = np.zeros((2, N))
    for k in range(N):
        outData[0][k] = Z[0][k]
        outData[1][k] = Z[1][k]
    for i in range(2):
        free(Z[i])
    free(Z)
    return outData