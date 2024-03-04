#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 13:16:51 2020

@author: andremarchildon
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
            self.maxeig_dExdq = lambda q : np.ones(q.shape)
            self.dExdq = lambda q : fn.diag(np.ones(q.shape))
            self.dExdq_eig_abs = self.dExdq
            self.central_Ex = self.central_fix_Ex
        
        if self.ay == self.ay_fix:
            print('Using the fixed ay={} diffeq functions since params match.'.format(self.ax_fix))
            self.maxeig_dEydq = lambda q : np.ones(q.shape)
            self.dEydq = lambda q : fn.diag(np.ones(q.shape))
            self.dEydq_eig_abs = self.dEydq
            self.central_Ey = self.central_fix_Ey

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
        
        #dExdq = np.array(self.a,ndmin=(q.ndim+1)) DO NOT USE!
        dExdq = fn.diag(np.ones(q.shape)*self.ax)
        return dExdq
    
    def dEydq(self, q):

        dFdq = fn.diag(np.ones(q.shape)*self.ay)
        return dFdq

    def dExdq_eig_abs(self, q):

        dExdq_eig_abs = fn.diag(np.ones(q.shape)*abs(self.ax))
        return dExdq_eig_abs
    
    def dEydq_eig_abs(self, q):

        dEydq_eig_abs = fn.diag(np.ones(q.shape)*abs(self.ay))
        return dEydq_eig_abs
    
    def maxeig_dExdq(self, q):
        ''' return the maximum eigenvalue - used for LF fluxes '''
        maxeig = np.ones(q.shape)*self.ax
        return maxeig

    def maxeig_dEydq(self, q):
        ''' return the maximum eigenvalue - used for LF fluxes '''
        maxeig = np.ones(q.shape)*self.ay
        return maxeig
    
    def d2Exdq2(self, q):

        dExdq = fn.diag(np.zeros(q.shape))
        return dExdq
    
    def d2Eydq2(self, q):

        dEydq = fn.diag(np.zeros(q.shape))
        return dEydq

    @njit   
    def central_fix_Ex(qL,qR):
        ''' a central 2-point flux for hadamard form but with ax fixed at 1.
        This allows us to jit the hadamard flux functions. '''
        f = fn.arith_mean(qL,qR)
        return f
    
    @njit   
    def central_fix_Ey(qL,qR):
        ''' a central 2-point flux for hadamard form but with ay fixed at 1.
        This allows us to jit the hadamard flux functions. '''
        f = fn.arith_mean(qL,qR)
        return f