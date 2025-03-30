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
    dim = 1
    neq_node = 1    # 1 equation in 1D
    pde_order = 1 # Order of the highest derivative in the PDE
    x = None
    has_exa_sol = True
    para_names = ('a',)
    a_fix = 1
    para_fix = [a_fix]

    def __init__(self, para, q0_type='SinWave'):

        super().__init__(para, q0_type)
        self.a = self.para[0]
        
        if self.a == self.a_fix:
            print('Using the fixed a={} diffeq functions since params match.'.format(self.a_fix))
            def dExdq_fix(q):
                nen,nelem = np.shape(q)
                return np.ones((nen,1,1,nelem),dtype=q.dtype)
            self.dExdq = dExdq_fix
            self.dExdq_abs = dExdq_fix
            self.maxeig_dExdq = lambda q : np.ones(q.shape)
            self.central_flux = self.central_fix_flux

    def exact_sol(self, time=0, x=None, guess=None):

        if x is None:
            x = self.x_elem

        x_mod = np.mod((x - self.xmin) - self.a*time, self.dom_len) + self.xmin
        exa_sol = self.set_q0(xy=x_mod)

        return exa_sol

    def dfdq(self, q):
        # I of course don't need this but it speeds things up a bit 
        # compared to the default function since it's so simple

        dfdq = - self.a * self.Dx
        return dfdq

    def calcEx(self, q):

        E = self.a * q
        return E
    
    def nonconservative_coeff(self, q):
        return self.a

    def dExdq(self, q):
        nen,nelem = np.shape(q)
        dExdq = self.a*np.ones((nen,1,1,nelem),dtype=q.dtype)
        return dExdq
    
    def dEndq(self, q, dxidx):
        nen,nelem = np.shape(q)
        dExdq = self.a*dxidx*np.ones((nen,1,1,nelem),dtype=q.dtype)
        return dExdq

    def d2Exdq2(self, q):
        nen,nelem = np.shape(q)
        dExdq = self.a*np.zeros((nen,1,1,nelem),dtype=q.dtype)
        return dExdq

    def dExdq_abs(self, q, entropy_fix):
        nen,nelem = np.shape(q)
        dExdq = abs(self.a)*np.ones((nen,1,1,nelem),dtype=q.dtype)
        return dExdq

    def maxeig_dExdq(self, q):
        ''' return the absolute maximum eigenvalue - used for LF fluxes '''
        maxeig = np.ones(q.shape)*abs(self.a)
        # this is actually slower: np.ones_like(qf_avg)*self.a
        return maxeig
    
    def maxeig_dEndq(self, q, dxidx):
        ''' return the absolute maximum eigenvalue - used for LF fluxes '''
        maxeig = abs(dxidx*self.a)
        return maxeig
    

    def calc_LF_const(self,xy):
        ''' Constant for the Lax-Friedrichs flux'''
        return abs(self.a)

    @njit   
    def central_fix_flux(qL,qR):
        ''' a central 2-point flux for hadamard form but with a fixed at 1.
        This allows us to jit the hadamard flux functions. '''
        f = fn.arith_mean(qL,qR)
        return f
