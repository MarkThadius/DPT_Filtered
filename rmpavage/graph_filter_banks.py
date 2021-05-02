# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 16:40:10 2019

@author: Mark
"""
#https://github.com/Ashish7129/Graph_Sampling
from rmpavage import rmpa
from PIL import Image
import numpy as np
import networkx as nx
import scipy.linalg as la
import scipy.signal as sig
import scipy.sparse as sparse
import copy as copy
import matplotlib.pyplot as plt
import time as time


class GraphFilter:
    # Initialise the Roadmakers Pavage process with a greyscale image.
    def __init__(self, R, K=None,w=None):
        if not (K or w):
            print('Either K, bandwidth, or w, cut-off frequency, must be defined for a filter to be constructed')
        
        self.R = R#reference operator
        self.K = K
        self.w = w
        
        t = time.process_time()
        elapsed_time = time.process_time() - t
        print("CPU time to initialize filter bank: ", elapsed_time)
        
        
    def cheby(self,w,i,Lmax,m=30,Lmin=0):
        M = m+1
        kernels = [lambda x: (x<=w), lambda x: (x > w)]
    
        a_arange = [Lmin,Lmax]
    
        a1 = (a_arange[1] - a_arange[0]) / 2
        a2 = (a_arange[1] + a_arange[0]) / 2
        c = np.zeros(m + 1)
    
        tmpN = np.arange(M)
        num = np.cos(np.pi * (tmpN + 0.5) / M)
        for o in range(m + 1):
            c[o] = 2. / M * np.dot(kernels[i](a1 * num + a2),np.cos(np.pi * o * (tmpN + 0.5) / M))
    
        return c
        
        
    def cheby_old(self,w,m,Lmax,i):
        M = m+1
        kernels = [lambda x: (x<=w), lambda x: (x > w)]
        c = np.zeros(m + 1)
        alpha = Lmax/2
        tmpM = np.arange(M) + 0.5
        num = np.cos(np.pi * (tmpM) / M)
        for o in range(m + 1):
            c[o] = 2. / M * np.dot(kernels[i](alpha * num + alpha),np.cos(np.pi * o * (tmpM) / M))
            
        return c
    
    def v(self,x):
#        print(x)
#        print(x[x>0])
        v = np.zeros(len(x))
#        print(len(x),len(v))
        v[x<=0] = 0
        v[(0<x)*(x<=1)] = 3*(x[(0<x)*(x<=1)]**2) - 2*(x[(0<x)*(x<=1)]**3)
        v[x>1] = 1        
        return v
    
    def smoothkernel(self,lam):
        return np.sqrt(self.v(2-(3/2*lam)))
    
    def cheby_meyer(self,w,m,Lmax):
        M = m+1
#        kernels = [lambda x: (x<=w), lambda x: (x > w)]
#        kernels = [lambda x: np.sqrt(self.v(2-(2/3*x)))]
        c = np.zeros([m + 1,2])
        alpha = Lmax/2
        tmpM = np.arange(M) + 0.5
        num = np.cos(np.pi * (tmpM) / M)
        for o in range(m + 1):
            c[o,0] = 2. / M * np.dot(self.smoothkernel(alpha * num + alpha),np.cos(np.pi * o * (tmpM) / M))
            c[o,1] = 2. / M * np.dot(self.smoothkernel(2-(alpha * num + alpha)),np.cos(np.pi * o * (tmpM) / M))
        return c
    
    
    def cheby_op(self,R, c, x,Lmax,Lmin=0):
        r"""
        Chebyshev polynomial of graph Laplacian applied to vector.
        Parameters
        ----------
        G : Graph
        c : ndarray or list of ndarrays
            Chebyshev coefficients for a Filter or a Filterbank
        signal : ndarray
            Signal to filter
        Returns
        -------
        r : ndarray
            Result of the filtering
        """
        M = len(c)    
        N = R.shape[0]    
        a_arange = [Lmin, Lmax]
    
        a1 = float(a_arange[1] - a_arange[0]) / 2.
        a2 = float(a_arange[1] + a_arange[0]) / 2.
    
        twf_old = x
        twf_cur = (R.dot(x) - a2 * x) / a1
        r = 0.5 * c[0] * twf_old + c[1] * twf_cur
    
        factor = 2/a1 * (R - a2 * sparse.eye(N))
        for k in range(2, M):
            twf_new = factor.dot(twf_cur) - twf_old
            r += c[k] * twf_new    
            twf_old = twf_cur
            twf_cur = twf_new

        return r
            
            
    def cheby_op_old(self,R, c, x, Lmax):
        r"""
        Chebyshev polynomial of graph Laplacian applied to vector.
        Parameters
        ----------
        R : Graph reference operator (A,L or Ln etc.)
        c : ndarray of Chebyshev coefficients for a specific Filter (low or high pass)
        x : ndarray of the Signal to filter
        Returns
        -------
        r : ndarray with the filtered signal
        """
    #    r = np.zeros(x.shape)
        alpha = Lmax/2
        M = len(c)
        N = R.shape[0]
    
        twf_old = x
        twf_cur = (R.dot(x) - alpha * x) / alpha
        r = 0.5 * c[0] * twf_old + c[1] * twf_cur
    
        factor = 2/alpha * (R - alpha * sparse.eye(N))
        for k in range(2, M):
            twf_new = factor.dot(twf_cur) - twf_old
            r += c[k] * twf_new    
            twf_old = twf_cur
            twf_cur = twf_new    
        return r
        
        




