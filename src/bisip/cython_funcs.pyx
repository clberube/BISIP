# -*- coding: utf-8 -*-
#!python
#cython: boundscheck=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: wraparound=False

"""
Created on Wed Nov  4 14:05:35 2015
@author:    charleslberube@gmail.com
            Polytechnique Montréal
Copyright (c) 2015-2016 Charles L. Bérubé
"""

import numpy as np
cimport numpy as cnp


DTYPE = np.float_
ctypedef cnp.float_t DTYPE_t

DTYPE2 = np.complex128
ctypedef cnp.complex128_t DTYPE2_t


cdef double complex jay
jay.real = 0.0
jay.imag = 1.0

cdef extern from "math.h":
    double exp(double x) nogil

cdef double complex C_ColeCole(double w_, double m_, double lt_, double c_):
    return m_*(1.0 - 1.0/(1.0 + ((jay*w_*exp(lt_))**c_)))

cdef double complex C_Dias(double w_, double R0_, double m_, double log_tau_, double eta_, double delta_):
    tau_p = exp(log_tau_)*(1/delta_ - 1)/(1 - m_)
    tau_pp = exp(log_tau_)**2 * eta_**2
    mu = jay*w_*exp(log_tau_) + (jay*w_*tau_pp)**0.5
    return R0_*(1 - m_*(1 - 1.0 / (1+jay*w_*tau_p*(1 + 1/mu))))

cdef double complex C_Shin(double w_, double R_, double log_Q_, double n_):
    z_cpe = 1 / (exp(log_Q_)*(jay*w_)**n_)
    return (1/z_cpe + 1/R_)**-1

cdef double complex C_Debye(double w_, double m_, double tau_, double c_):
    return m_*(1 - 1.0/(1 + ((jay*w_*(tau_))**c_)))

def ColeCole_cyth(cnp.ndarray[DTYPE_t, ndim=1] w, DTYPE_t R0, cnp.ndarray[DTYPE_t, ndim=1] m, cnp.ndarray[DTYPE_t, ndim=1] lt, cnp.ndarray[DTYPE_t, ndim=1] c):
    cdef int N = w.shape[0]
    cdef int D = m.shape[0]
    cdef int i, j
    cdef double complex z_
    cdef cnp.ndarray[DTYPE_t, ndim=2] Z = np.empty((2,N), dtype=DTYPE)
    for j in range(N):
        z_ = 0
        for i in range(D):
            z_ += C_ColeCole(w[j], m[i], lt[i], c[i])
        z_ = R0*(1 - z_)
        Z[0,j] = z_.real
        Z[1,j] = z_.imag
    return Z

def Dias2000_cyth(cnp.ndarray[DTYPE_t, ndim=1] w, DTYPE_t R0, DTYPE_t m, DTYPE_t log_tau, DTYPE_t eta, DTYPE_t delta):
    cdef int N = w.shape[0]
    cdef int j
    cdef cnp.ndarray[DTYPE_t, ndim=2] Z = np.empty((2,N), dtype=DTYPE)
    cdef cnp.ndarray[DTYPE2_t, ndim=1] z_ = np.zeros(N, dtype=DTYPE2)
    for j in range(N):
        z_[j] = C_Dias(w[j], R0, m, log_tau, eta, delta)
        Z[0,j] = z_[j].real
        Z[1,j] = z_[j].imag
    return Z

def Decomp_cyth(cnp.ndarray[DTYPE_t, ndim=1] w, cnp.ndarray[DTYPE_t, ndim=1] taus, cnp.ndarray[DTYPE_t, ndim=2] log_taus, DTYPE_t c_exp, DTYPE_t R0, cnp.ndarray[DTYPE_t, ndim=1] a):
    cdef int D = a.shape[0]
    cdef int N = w.shape[0]
    cdef int S = taus.shape[0]
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
            z_ +=  C_Debye(w[j], M[k], taus[k], c_exp)
        z_ = R0*(1 - z_)
        Z[0,j] = z_.real
        Z[1,j] = z_.imag
    return Z

def Shin2015_cyth(cnp.ndarray[DTYPE_t, ndim=1] w, cnp.ndarray[DTYPE_t, ndim=1] R, cnp.ndarray[DTYPE_t, ndim=1] log_Q, cnp.ndarray[DTYPE_t, ndim=1] n):
    cdef int D = R.shape[0]
    cdef int N = w.shape[0]
    cdef int i, j
    cdef double complex z_
    cdef cnp.ndarray[DTYPE_t, ndim=2] Z = np.empty((2, N), dtype=DTYPE)
    for j in range(N):
        z_ = 0
        for i in range(D):
            z_ += C_Shin(w[j], R[i], log_Q[i], n[i])
        Z[0,j] = z_.real
        Z[1,j] = z_.imag
    return Z
