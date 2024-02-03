#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 12:41:26 2020

@author: andremarchildon
"""

import numpy as np

from Source.DiffEq.DiffEqBase import PdeBaseCons
import Source.Methods.Functions as fn
from numba import njit

class Burgers(PdeBaseCons):
    '''
    Purpose
    ----------
    This class provides the required functions to solve the Burgers equation
    '''

    # Diffeq info
    diffeq_name = 'Burgers'
    dim = 1
    npar = 0        # No. of design parameters
    neq_node = 1    # 1 equation in 1D
    eq_type = 'pde'
    pde_order = 1

    def __init__(self, para=None, q0_type='SinWave',
                 use_split_form=False, split_alpha=2/3):

        super().__init__(para, q0_type)
        self.use_split_form = use_split_form
        self.split_alpha = split_alpha

    def calcEx(self, q):
        E = 0.5*q**2
        return E

    def dExdx(self, q):

        if self.use_split_form:
            dExdx = (self.split_alpha/2.)*fn.gm_gv(self.Dx, q**2) + (1.-self.split_alpha)*fn.gdiag_gv(q,fn.gm_gv(self.Dx, q))
        else:
            E = self.calcEx(q)
            dExdx = fn.gm_gv(self.Dx, E)

        return dExdx

    def dExdq(self, q):

        dExdq = fn.diag(q)
        return dExdq
    
    def d2Exdq2(self, q):
        
        d2Exdq2 = fn.diag(np.ones(q.shape))
        return d2Exdq2

    def dfdq(self, q):
        # take dExdx as a vector a_i(q) and find matrix d(a_i)/d(q_j)

        if self.use_split_form:
            # these both do the same, but the second is a bit faster
            #dfdq = -(1/3)*(2*fn.lm_gm(self.der1,fn.diag(q)) + fn.diag(self.der1@q) + fn.gm_lm(fn.diag(q),self.der1))
            #dfdq = -(1/3)*(2*np.multiply(self.der1[:,:,None],q) + fn.diag(self.der1 @ q) + fn.gm_lm(fn.diag(q),self.der1))
            dfdq = -self.split_alpha*fn.gm_gv_colmultiply(self.Dx,q) - (1-self.split_alpha)*(fn.diag(fn.gm_gv(self.Dx, q)) + fn.gdiag_gm(q,self.Dx))
        else:
            # this does the same as the base function, just a bit faster
            dfdq = - fn.gm_gv_colmultiply(self.Dx,q)
            
        return dfdq

    def calc_LF_const(self):
        ''' Constant for the Lax-Friedrichs flux'''
        q = fn.check_q_shape(self.set_q0())
        return np.max(np.abs(q))

    def dExdq_eig_abs(self, dExdq):

        dExdq_eig_abs = abs(dExdq)
        return dExdq_eig_abs
    
    def maxeig_dExdq(self, q):
        ''' return the maximum eigenvalue - used for LF fluxes '''
        return np.abs(q)
    
    def entropy(self,q):
        e = q**2/2
        return e
    
    def a_energy(self,q):
        ''' compute the global U-norm SBP energy of global solution vector q '''
        return np.tensordot(q, q * self.H * q)
    
    def a_energy_der(self,q,dqdt):
        ''' compute the global U-norm SBP energy derivatve of global solution vector q '''
        return 2 * np.tensordot(q, q * self.H * dqdt)
    
    def a_conservation(self,q):
        ''' compute the global A-conservation SBP of global solution vector q, equal to entropy/energy '''
        return np.sum(q * self.H * q)
    
    @njit
    def ec_Ex(qL,qR):
        ''' entropy conservative 2-point flux for the Hadamard form '''
        fx = (qL**2 + qL*qR + qR**2)/6
        return fx
    
    @njit   
    def central_fix_Ex(qL,qR):
        ''' a central 2-point flux for hadamard form.
        This allows us to jit the hadamard flux functions. '''
        f = fn.arith_mean(qL**2/2,qR**2/2)
        return f

