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
cimport numpy as cnp

from libc.stdlib cimport malloc, free

DTYPE = np.float_
ctypedef cnp.float_t DTYPE_t
DTYPE2 = np.complex128
ctypedef cnp.complex128_t DTYPE2_t

cdef double complex C_ColeCole(double w_, double m_, double lt_, double c_):
    cdef double complex Z_i
    cdef double complex jay
    jay.real = 0.0
    jay.imag = 1.0
    Z_i = m_*(1.0 - 1.0/(1.0 + ((jay*w_*(10.0**lt_))**c_)))
    return Z_i

cdef double complex C_Dias(double w_, double R0_, double m_, double log_tau_, double eta_,double delta_):
    cdef double complex Z_i
    cdef double complex jay
    jay.real = 0.0
    jay.imag = 1.0
    Z_i = R0_*(1 - (m_*(1 - 1.0/(1 + (jay*w_*(((10**log_tau_)/delta_)*(1 - delta_)/(1 - m_)))*(1 + 1.0/(jay*w_*(10**log_tau_) + eta_*(10**log_tau_)*(jay*w_**0.5)))))))
    return Z_i

cdef double complex C_Shin(double w_, double R_, double log_Q_, double n_):
    cdef double complex Z_i
    cdef double complex jay
    jay.real = 0.0
    jay.imag = 1.0
    Z_i = R_ / ((jay*w_**n_)*((10**log_Q_))*R_ + 1)
    return Z_i

cdef double complex C_Debye(double w_, double m_, double tau_, double c_):
    cdef double complex Z_i
    cdef double complex jay
    jay.real = 0.0
    jay.imag = 1.0
    Z_i = m_*(1 - 1.0/(1 + ((jay*w_*(tau_))**c_)))
    return Z_i

def ColeCole_cyth1(cnp.ndarray[DTYPE_t, ndim=1] w, DTYPE_t R0, cnp.ndarray[DTYPE_t, ndim=1] m, cnp.ndarray[DTYPE_t, ndim=1] lt, cnp.ndarray[DTYPE_t, ndim=1] c):
    cdef int N = w.shape[0]
    cdef int D = m.shape[0]
    cdef int i, j
    cdef double complex z_
#    cdef double zr = z_.real
#    cdef double zi = z_.imag
    cdef cnp.ndarray[DTYPE_t, ndim=2] Z = np.empty((2,N), dtype=DTYPE)
    for j in range(N):
        z_ = 0
        for i in range(D):
            z_ += C_ColeCole(w[j], m[i], lt[i], c[i])
        z_ = R0*(1 - z_)
        Z[0,j] = z_.real
        Z[1,j] = z_.imag
    return Z

def ColeCole_cyth2(double[:] w, double R0, double[:] m, double[:] lt, double[:] c):
    cdef int N = w.shape[0]
    cdef int D = m.shape[0]
    cdef int i, j
    cdef int rows = 2
    cdef double complex z_
    cdef double[:,:] Z = np.empty((rows,N), dtype=DTYPE)
    for j in range(N):
        z_ = 0
        for i in range(D):
            z_ += C_ColeCole(w[j], m[i], lt[i], c[i])
        z_ = R0*(1.0 - z_)
        Z[0,j] = z_.real
        Z[1,j] = z_.imag
    return Z

def Dias_cyth(cnp.ndarray[DTYPE_t, ndim=1] w, DTYPE_t R0, DTYPE_t m, DTYPE_t log_tau, DTYPE_t eta, DTYPE_t delta):
    cdef int N = w.shape[0]
    cdef int j
    cdef cnp.ndarray[DTYPE_t, ndim=2] Z = np.empty((2,N), dtype=DTYPE)
    cdef cnp.ndarray[DTYPE2_t, ndim=1] z_ = np.zeros(N, dtype=DTYPE2)
    for j in range(N):
        z_[j] = C_Dias(w[j], R0, m, log_tau, eta, delta)
        Z[0,j] = z_[j].real
        Z[1,j] = z_[j].imag
    return Z

def Decomp_cyth(cnp.ndarray[DTYPE_t, ndim=1] w, cnp.ndarray[DTYPE_t, ndim=1] tau_10, cnp.ndarray[DTYPE_t, ndim=2] log_taus, DTYPE_t c_exp, DTYPE_t R0, cnp.ndarray[DTYPE_t, ndim=1] a):
    cdef int D = a.shape[0]
    cdef int N = w.shape[0]
    cdef int S = tau_10.shape[0]
    cdef int i, j, k
    cdef cnp.ndarray[DTYPE_t, ndim=1] M = np.zeros(S, dtype=DTYPE)
    cdef double complex z_
    cdef double complex z_hi
    cdef cnp.ndarray[DTYPE_t, ndim=2] Z = np.empty((2,N), dtype=DTYPE)
    for i in range(D):
        for k in range(S):
            M[k] = M[k] + a[i]*(log_taus[i,k])    
    for j in range(N):
        z_ = 0 
        for k in range(S):
            z_ +=  C_Debye(w[j], M[k], tau_10[k], c_exp)
#        z_hi = C_Debye(w[j], m_hi, 10**log_tau_hi, 1.0)
#        z_ += z_hi
        z_ = R0*(1 - z_)
        Z[0,j] = z_.real
        Z[1,j] = z_.imag
    return Z

def Shin_cyth(cnp.ndarray[DTYPE_t, ndim=1] w, cnp.ndarray[DTYPE_t, ndim=1] R, cnp.ndarray[DTYPE_t, ndim=1] log_Q, cnp.ndarray[DTYPE_t, ndim=1] n):
    cdef int D = R.shape[0]
    cdef int N = w.shape[0]
    cdef int i, j
    cdef cnp.ndarray[DTYPE2_t, ndim=1] z_ = np.zeros(N, dtype=DTYPE2)
    cdef cnp.ndarray[DTYPE_t, ndim=2] Z = np.empty((2,N), dtype=DTYPE)
    for j in range(N):
        for i in range(D):
            z_[j] = z_[j] + C_Shin(w[j], R[i], log_Q[i], n[i])
        Z[0,j] = z_[j].real
        Z[1,j] = z_[j].imag
    return Z