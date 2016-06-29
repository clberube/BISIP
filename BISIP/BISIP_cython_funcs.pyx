# -*- coding: utf-8 -*-
#!python
#cython: boundscheck=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: wraparound=False
#cython: boundscheck=False

"""
Created on Wed Nov  4 14:05:35 2015

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
    Z_i = m_*(1 - 1.0/(1 + (powc((jay*w_*(10**lt_)),c_))))
    return Z_i

cdef double complex C_Dias(double w_, double R0_, double m_, double log_tau_, double eta_,double delta_):
    cdef double complex Z_i
    cdef double complex jay
    jay.real = 0.0
    jay.imag = 1.0
    Z_i = R0_*(1 - (m_*(1 - 1.0/(1 + (jay*w_*(((10**log_tau_)/delta_)*(1 - delta_)/(1 - m_)))*(1 + 1.0/(jay*w_*(10**log_tau_) + eta_*(10**log_tau_)*powc(jay*w_,0.5)))))))
    return Z_i

cdef double complex C_Shin(double w_, double R_, double log_Q_, double n_):
    cdef double complex Z_i
    cdef double complex jay
    jay.real = 0.0
    jay.imag = 1.0
    Z_i = R_ / (powc(jay*w_,n_)*((10**log_Q_))*R_ + 1)
    return Z_i

cdef double complex C_Debye(double w_, double tau_, double m_):
    cdef double complex Z_i
    Z_i = m_*(1 - 1.0/(1+1j*w_*tau_))
    return Z_i

def m_cyth(np.ndarray[DTYPE_t, ndim=1] mp):
    cdef int D = mp.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=1] m = np.zeros(D, dtype=DTYPE)
    for i in range(D):
        m[i] = 1.0/( (1.0 / 10**mp[i]) + 1 )
    return m

def ColeCole_cyth(np.ndarray[DTYPE_t, ndim=1] w, DTYPE_t R0, np.ndarray[DTYPE_t, ndim=1] m, np.ndarray[DTYPE_t, ndim=1] lt, np.ndarray[DTYPE_t, ndim=1] c):
    cdef int N = w.shape[0]
    cdef int D = m.shape[0]
    cdef int i, j
    cdef np.ndarray[DTYPE2_t, ndim=1] z_ = np.zeros(N, dtype=DTYPE2)
    cdef np.ndarray[DTYPE_t, ndim=2] Z = np.empty((2,N), dtype=DTYPE)
    for j in range(N):
        for i in range(D):
            z_[j] = z_[j] + C_ColeCole(w[j], m[i], lt[i], c[i])
        Z[0,j] = R0*(1 - z_[j]).real
        Z[1,j] = R0*(1 - z_[j]).imag
    return Z

def Dias_cyth(np.ndarray[DTYPE_t, ndim=1] w, DTYPE_t R0, DTYPE_t m, DTYPE_t log_tau, DTYPE_t eta, DTYPE_t delta):
    cdef int N = w.shape[0]
    cdef int j
    cdef np.ndarray[DTYPE_t, ndim=2] Z = np.empty((2,N), dtype=DTYPE)
    cdef np.ndarray[DTYPE2_t, ndim=1] z_ = np.zeros(N, dtype=DTYPE2)
    for j in range(N):
        z_[j] = C_Dias(w[j], R0, m, log_tau, eta, delta)
        Z[0,j] = z_[j].real
        Z[1,j] = z_[j].imag
    return Z

def Debye_cyth(np.ndarray[DTYPE_t, ndim=1] w, np.ndarray[DTYPE_t, ndim=1] tau_10, np.ndarray[DTYPE_t, ndim=2] log_taus, DTYPE_t log_tau_hi, DTYPE_t m_hi, DTYPE_t R0, np.ndarray[DTYPE_t, ndim=1] a):
    cdef int D = a.shape[0]
    cdef int N = w.shape[0]
    cdef int S = tau_10.shape[0]
    cdef int i, j, k
    cdef np.ndarray[DTYPE_t, ndim=1] M = np.zeros(S, dtype=DTYPE)
    cdef np.ndarray[DTYPE2_t, ndim=1] z_hi = np.empty(N, dtype=DTYPE2)
    cdef np.ndarray[DTYPE2_t, ndim=1] z_de = np.zeros(S, dtype=DTYPE2)
    cdef np.ndarray[DTYPE2_t, ndim=1] z_ = np.zeros(N, dtype=DTYPE2)
    cdef np.ndarray[DTYPE_t, ndim=2] Z = np.empty((2,N), dtype=DTYPE)
    for i in range(D):
        for k in range(S):
            M[k] = M[k] + a[i]*(log_taus[i,k])
    for j in range(N):
        for k in range(S):
            z_de[j] = z_de[j] + M[k]*(1 - 1.0/(1+1j*w[j]*tau_10[k]))
        z_hi[j] = m_hi*(1 - 1.0/(1+1j*w[j]*((10**log_tau_hi))))
        z_[j] = R0*(1 - (z_hi[j] + z_de[j]))
        Z[0,j] = z_[j].real
        Z[1,j] = z_[j].imag
    return Z

def Debye_cyth2(np.ndarray[DTYPE_t, ndim=1] w, np.ndarray[DTYPE_t, ndim=1] tau, DTYPE_t R0, np.ndarray[DTYPE_t, ndim=1] m):
    cdef int N = w.shape[0]
    cdef int S = tau.shape[0]
    cdef int i, j
    cdef np.ndarray[DTYPE2_t, ndim=1] z_de = np.zeros(S, dtype=DTYPE2)
    cdef np.ndarray[DTYPE2_t, ndim=1] z_ = np.zeros(N, dtype=DTYPE2)
    cdef np.ndarray[DTYPE_t, ndim=2] Z = np.empty((2,N), dtype=DTYPE)
    for j in range(N):
        for i in range(S):
            z_de[j] = z_de[j] + C_Debye(w[j], tau[i], m[i])
        z_[j] = R0*(1 - z_de[j])
        Z[0,j] = z_[j].real
        Z[1,j] = z_[j].imag
    return Z

def Shin_cyth(np.ndarray[DTYPE_t, ndim=1] w, np.ndarray[DTYPE_t, ndim=1] R, np.ndarray[DTYPE_t, ndim=1] log_Q, np.ndarray[DTYPE_t, ndim=1] n):
    cdef int D = R.shape[0]
    cdef int N = w.shape[0]
    cdef int i, j
    cdef np.ndarray[DTYPE2_t, ndim=1] z_ = np.zeros(N, dtype=DTYPE2)
    cdef np.ndarray[DTYPE_t, ndim=2] Z = np.empty((2,N), dtype=DTYPE)
    for j in range(N):
        for i in range(D):
            z_[j] = z_[j] + C_Shin(w[j], R[i], log_Q[i], n[i])
        Z[0,j] = z_[j].real
        Z[1,j] = z_[j].imag
    return Z