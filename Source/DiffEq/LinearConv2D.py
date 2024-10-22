#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 13:16:51 2020

@author: bercik
"""

import numpy as np

from Source.DiffEq.DiffEqBase import PdeBase
import Source.Methods.Functions as fn
from numba import njit


class LinearConv(PdeBase):
    '''
    Purpose
    ----------
    This class provides the required functions to solve the linear convection
    equation:
    '''

    diffeq_name = 'LinearConvection'
    dim = 2
    neq_node = 1    # 1 equation in 1D
    pde_order = 1 # Order of the highest derivative in the PDE
    xy = None
    has_exa_sol = True
    para_names = ('ax','ay')
    ax_fix = 1
    ay_fix = 1
    para_fix = [ax_fix,ay_fix]

    def __init__(self, para=None, q0_type='SinWave'):

        super().__init__(para, q0_type)
        self.ax = self.para[0]
        self.ay = self.para[1]
        
        if self.ax == self.ax_fix:
            print('Using the fixed ax={} diffeq functions since params match.'.format(self.ax_fix))
            def dExdq_fix(q):
                nen,nelem = np.shape(q)
                return np.ones((nen,1,1,nelem),dtype=q.dtype)
            self.dExdq = dExdq_fix
            self.dExdq_abs = dExdq_fix
            self.maxeig_dExdq = lambda q : np.ones(q.shape)
        
        if self.ay == self.ay_fix:
            print('Using the fixed ay={} diffeq functions since params match.'.format(self.ay_fix))
            def dEydq_fix(q):
                nen,nelem = np.shape(q)
                return np.ones((nen,1,1,nelem),dtype=q.dtype)
            self.dEydq = dEydq_fix
            self.dEydq_abs = dEydq_fix
            self.maxeig_dEydq = lambda q : np.ones(q.shape)

        if self.ax == self.ax_fix and self.ay == self.ay_fix:
            self.central_fluxes = self.central_fix_fluxes

    def exact_sol(self, time=0, xy=None):

        if xy is None:
            xy = self.xy_elem
        xy_mod = np.empty(xy.shape)

        xy_mod[:,0,:] = np.mod((xy[:,0,:] - self.xmin[0]) - self.ax*time, self.dom_len[0]) + self.xmin[0]
        xy_mod[:,1,:] = np.mod((xy[:,1,:] - self.xmin[1]) - self.ay*time, self.dom_len[1]) + self.xmin[1]
        exa_sol = self.set_q0(xy=xy_mod)

        return exa_sol

    def calcEx(self, q):

        E = self.ax * q
        return E
    
    def calcEy(self, q):

        F = self.ay * q
        return F

    def dExdq(self, q):
        nen,nelem = np.shape(q)
        dExdq = self.ax*np.ones((nen,1,1,nelem),dtype=q.dtype)
        return dExdq
    
    def dEydq(self, q):
        nen,nelem = np.shape(q)
        dEydq = self.ay*np.ones((nen,1,1,nelem),dtype=q.dtype)
        return dEydq

    def dExdq_abs(self, q, entropy_fix):
        nen,nelem = np.shape(q)
        dExdq = abs(self.ax)*np.ones((nen,1,1,nelem),dtype=q.dtype)
        return dExdq
    
    def dEydq_abs(self, q, entropy_fix):
        nen,nelem = np.shape(q)
        dEydq = abs(self.ay)*np.ones((nen,1,1,nelem),dtype=q.dtype)
        return dEydq
    
    def maxeig_dExdq(self, q):
        ''' return the maximum eigenvalue - used for LF fluxes '''
        maxeig = np.ones(q.shape)*self.ax
        return maxeig

    def maxeig_dEydq(self, q):
        ''' return the maximum eigenvalue - used for LF fluxes '''
        maxeig = np.ones(q.shape)*self.ay
        return maxeig
    
    def maxeig_dEndq(self, q, metrics):
        ''' return the maximum eigenvalue - used for LF fluxes '''
        maxeig = np.abs(metrics[:,0,:]*self.ax + metrics[:,1,:]*self.ay)
        return maxeig
    
    def d2Exdq2(self, q):
        nen,nelem = np.shape(q)
        dExdq = np.zeros((nen,1,1,nelem),dtype=q.dtype)
        return dExdq
    
    def d2Eydq2(self, q):
        nen,nelem = np.shape(q)
        dEydq = np.zeros((nen,1,1,nelem),dtype=q.dtype)
        return dEydq

    @njit
    def central_fix_fluxes(qL,qR):
        ''' a central 2-point flux for hadamard form but with ax fixed at (1,1).
        This allows us to jit the hadamard flux functions. '''
        f = fn.arith_mean(qL,qR)
        return f, f