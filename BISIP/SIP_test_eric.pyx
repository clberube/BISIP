# -*- coding: utf-8 -*-
#!python
#cython: boundscheck=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: wraparound=False
#cython: boundscheck=False

"""
Created on Wed Nov 30 13:04:50 2016

@author: Charles
"""

import numpy as np
cimport numpy as np

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



def ColeCole_cyth(np.ndarray[DTYPE_t, ndim=1] w, DTYPE_t R0, np.ndarray[DTYPE_t, ndim=1] m, np.ndarray[DTYPE_t, ndim=1] lt, np.ndarray[DTYPE_t, ndim=1] c):
    cdef int N = w.shape[0]
    cdef int D = m.shape[0]
    cdef int i, j     
    
    Z = <double **>malloc(2 * sizeof(double*))  
    for i in range(nld1):
        Z[i] = <double *>malloc(N * sizeof(double))   
        
    for j in range(N):
        for i in range(D):
            z_[j] = z_[j] + C_ColeCole(w[j], m[i], lt[i], c[i])
        Z[0,j] = R0*(1 - z_[j]).real
        Z[1,j] = R0*(1 - z_[j]).imag
        
        
    try:
        return Z
    
    finally:
        for i in range(2):
            Z[i] = free  
        free(Z) 