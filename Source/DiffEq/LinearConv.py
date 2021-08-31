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
    dim = 1
    neq_node = 1    # 1 equation in 1D
    npar = 0        # No. of design parameters
    pde_order = 1 # Order of the highest derivative in the PDE
    x = None
    has_exa_sol = True
    para_names = ('a',)

    def __init__(self, para, obj_name=None, q0_type='SinWave'):

        super().__init__(para, obj_name, q0_type)
        self.a = self.para[0]

    def exact_sol(self, time=0):

        assert self.dim==1, 'exact sol only setup for 1D'

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

    def dEdq(self, q):
        
        dEdq = fn.diag(np.ones(q.shape)*self.a)
        return dEdq

    def d2Edq2(self, q):

        dEdq = fn.diag(np.zeros(q.shape))
        return dEdq

    def dEdq_eig_abs(self, dEdq):

        dEdq_eig_abs = np.abs(dEdq)
        return dEdq_eig_abs
    
    def maxeig_dEdq(self, q):
        ''' return the maximum eigenvalue - used for LF fluxes '''
        maxeig = np.ones(q.shape)*self.a
        return maxeig

    def calc_LF_const(self,xy):
        ''' Constant for the Lax-Friedrichs flux'''
        return abs(self.a)

    @njit   
    def central_fix_Ex(qL,qR):
        ''' a central 2-point flux for hadamard form but with a fixed at 1.
        This allows us to jit the hadamard flux functions. '''
        f = fn.arith_mean(qL,qR)
        return f
