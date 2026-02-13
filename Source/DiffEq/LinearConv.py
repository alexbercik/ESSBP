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
    pde_order1 = True
    pde_order2 = False
    x = None
    has_exa_sol = True
    para_names = ('a',)
    a_fix = 1
    para_fix = [a_fix]

    def __init__(self, para, q0_type='SinWave', had_flux='central'):

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
            self.logarithmic_flux = self.logarithmic_fix_flux
            self.geometric_flux = self.geometric_fix_flux
        
        if had_flux.lower() == 'central':
            self.entropy = self.entropy_central
            self.entropy_var = self.entropy_var_central
            self.dqdw = self.dqdw_central
            self.maxeig_dqdw = self.maxeig_dqdw_central
        elif had_flux.lower() == 'logarithmic':
            self.entropy = self.entropy_log
            self.entropy_var = self.entropy_var_log
            self.dqdw = self.dqdw_log
            self.maxeig_dqdw = self.maxeig_dqdw_log
        elif had_flux.lower() == 'geometric':
            self.entropy = self.entropy_geom
            self.entropy_var = self.entropy_var_geom
            self.dqdw = self.dqdw_geom
            self.maxeig_dqdw = self.maxeig_dqdw_geom
        else:
            raise ValueError(f'Invalid had_flux: {had_flux}. Must be "central", "logarithmic", or "geometric".')

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

    @njit   
    def logarithmic_fix_flux(qL,qR):
        ''' a logarithmic 2-point flux for hadamard form but with a fixed at 1.
        This allows us to jit the hadamard flux functions. '''
        f = fn.log_mean(qL,qR)
        return f

    @njit   
    def geometric_fix_flux(qL,qR):
        ''' a geometric 2-point flux for hadamard form but with a fixed at 1.
        This allows us to jit the hadamard flux functions. '''
        f = fn.geom_mean(qL,qR)
        return f

    def entropy_central(self, q):
        ''' nodal values of the entropy for the central flux '''
        return q*q
    
    def entropy_var_central(self, q):
        ''' nodal values of the entropy variables w(q) for the central flux '''
        return q

    def dqdw_central(self,q):
        ''' hessian P of potential phi wrt entropy variables w '''
        return fn.gdiag_to_gbdiag(np.ones_like(q))
    
    def maxeig_dqdw_central(self,q):
        ''' maximum eigenvalues of hessian P, i.e. abs(dqdw) for scalar '''
        return np.ones(q.shape)

    def entropy_geom(self, q):
        ''' nodal values of the entropy for the central flux '''
        return -2*np.sqrt(q)
    
    def entropy_var_geom(self, q):
        ''' nodal values of the entropy variables w(q) for the central flux '''
        return -1/np.sqrt(q)

    def dqdw_geom(self,q):
        ''' hessian P of potential phi wrt entropy variables w '''
        return fn.gdiag_to_gbdiag(2*q*np.sqrt(q))
    
    def maxeig_dqdw_geom(self,q):
        ''' maximum eigenvalues of hessian P, i.e. abs(dqdw) for scalar '''
        return 2*np.abs(q*np.sqrt(q))

    def entropy_log(self, q):
        ''' nodal values of the entropy for the central flux '''
        return q*np.log(q) - q
    
    def entropy_var_log(self, q):
        ''' nodal values of the entropy variables w(q) for the central flux '''
        return np.log(q)

    def dqdw_log(self,q):
        ''' hessian P of potential phi wrt entropy variables w '''
        return fn.gdiag_to_gbdiag(q)
    
    def maxeig_dqdw_log(self,q):
        ''' maximum eigenvalues of hessian P, i.e. abs(dqdw) for scalar '''
        return np.abs(q)