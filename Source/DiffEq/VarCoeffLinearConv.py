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
    This class provides the required functions to solve the variable coefficient
    linear convection equation:
    '''

    diffeq_name = 'VariableCoefficientLinearConvection'
    dim = 1
    neq_node = 1    # 1 equation in 1D
    npar = 0        # No. of design parameters
    pde_order1 = True
    pde_order2 = False
    x = None
    has_exa_sol = False # TODO: can I put in an exact solution?
    para_names = ('alpha',)
    use_exact_der = True # whether to compute variable coefficient derivative exactly
    extrapolate_bdy_flux = True # whether to extrapolate fluxes or use extrapolated solution in SAT

    def __init__(self, para, q0_type='SinWave', a_type='Gaussian'):

        super().__init__(para, q0_type)
        self.alpha = self.para[0] # split form parameter
        self.a_type = a_type
        self.a = None # to be set later
        self.ader = None
            
    def afun(self, x):
        if self.a_type == 'Gaussian':
            q0_max_q = self.q0_max_q/2
            k = (8*np.log(self.q0_gauss_wave_val_bc/q0_max_q))
            mid_point = 0.5*(self.xmax + self.xmin) # mean
            stdev2 = abs((self.xmax - self.xmin)**2/k) # standard deviation squared
            exp = -0.5*(x-mid_point)**2/stdev2
            a = q0_max_q * np.exp(exp)
        elif self.a_type == 'shifted Gaussian':
            q0_max_q = self.q0_max_q/2
            k = (8*np.log(self.q0_gauss_wave_val_bc/q0_max_q))
            mid_point = 0.5*(self.xmax + self.xmin) # mean
            stdev2 = abs((self.xmax - self.xmin)**2/k) # standard deviation squared
            exp = -0.5*(x-mid_point)**2/stdev2
            a = q0_max_q * np.exp(exp) + 1
        elif self.a_type == 'constant':
            a = np.ones(np.shape(x)) 
        elif self.a_type == 'linear':
            mid = 1
            a = x/(self.xmin-self.xmax) + (self.xmin+self.xmax)/(self.xmax-self.xmin) + mid
        elif self.a_type == 'linear_eps':
            mid = 0.5 + 1e-8
            a = x/(self.xmin-self.xmax) + (self.xmin+self.xmax)/(self.xmax-self.xmin) + mid
        elif self.a_type == 'linear_0':
            mid = 0.5
            a = x/(self.xmin-self.xmax) + (self.xmin+self.xmax)/(self.xmax-self.xmin) + mid
        elif self.a_type == 'linear_neg':
            mid = 0
            a = x/(self.xmin-self.xmax) + (self.xmin+self.xmax)/(self.xmax-self.xmin) + mid
        else:
            raise Exception('Variable coefficient not understood.')
        return a
    
    def afunder(self, x):
        if self.a_type == 'Gaussian' or self.a_type == 'shifted Gaussian':
            q0_max_q = self.q0_max_q/2
            k = (8*np.log(self.q0_gauss_wave_val_bc/q0_max_q))
            mid_point = 0.5*(self.xmax + self.xmin) # mean
            stdev2 = abs((self.xmax - self.xmin)**2/k) # standard deviation squared
            exp = -0.5*(x-mid_point)**2/stdev2
            ader = - q0_max_q * np.exp(exp) * (x-mid_point) / stdev2
        elif self.a_type == 'constant':
            ader = np.zeros(np.shape(x))
        elif 'linear' in self.a_type:
            ader = np.ones(np.shape(x))/(self.xmin-self.xmax)
        else:
            raise Exception('Variable coefficient not understood.')
        return ader

    def exact_sol(self, time=0, x=None, guess=None):
        
        if x is None:
            x = self.x_elem
        # TODO

        return None

    def dfdq(self, q):
        # TODO
        
        return None

    def calcEx(self, q):

        E = self.a * q
        return E
    
    def dExdx(self, q):
        ''' Overwrites default divergence form with a potentially split form '''
        
        E = self.calcEx(q)
        
        if self.use_exact_der:      
            dExdx = self.alpha * fn.gm_gv(self.Dx, E) + \
                (1 - self.alpha) * ( self.a * fn.gm_gv(self.Dx, q) + q * self.ader )
        else:
            dExdx = self.alpha * fn.gm_gv(self.Dx, E) + \
                (1 - self.alpha) * ( self.a * fn.gm_gv(self.Dx, q) + \
                                          q * fn.gm_gv(self.Dx, self.a) )
        return dExdx

    def dExdq(self, q):
        dExdq = np.reshape(self.a, (self.nen,1,1,self.nelem))
        return dExdq
    
    def d2Exdq2(self, q):

        d2Exdq2 = np.zeros(q.shape)
        return d2Exdq2
    
    def dExdq_abs(self, q, entropy_fix):
        dExdq = np.reshape(np.abs(self.a), (self.nen,1,1,self.nelem))
        return dExdq
    
    def maxeig_dExdq(self, q):
        ''' return the maximum eigenvalue - used for LF fluxes '''
        # Note: Because I don't have access to x (usually q is actually a q_facet)
        # I can not do a local LF. Instead I return an overly dissipative global LF
        return np.ones(q.shape)*np.max(self.a)

    def calc_LF_const(self,xy,use_local=False):
        ''' Constant for the Lax-Friedrichs flux'''
        if use_local:
            c = np.max(abs(self.a),axis=0)
        else:
            c = np.max(abs(self.a))
        return c
    
    def a_energy(self,q):
        ''' compute the global A-norm SBP energy of global solution vector q '''
        return np.tensordot(q, self.a * self.H * q)
    
    def a_energy_der(self,q,dqdt):
        ''' compute the global A-norm SBP energy derivatve of global solution vector q '''
        return 2 * np.tensordot(q, self.a * self.H * dqdt)
    
    def a_cons(self,q):
        ''' compute the global A-conservation SBP of global solution vector q '''
        return np.sum(self.a * self.H * q)
    
    def set_mesh(self, mesh):
        '''
        Purpose
        ----------
        Needed to calculate the initial solution and to calculate source terms,
        must overwrite base case so we can set the variable coefficient
        '''

        self.mesh = mesh

        ''' Extract other parameters '''
        assert self.dim == self.mesh.dim,'Dimensions of DiffEq and Solver do not match.'

        self.x = self.mesh.x
        self.x_elem = self.mesh.x_elem
        self.x_ref = self.mesh.x_op
        #self.dx = self.mesh.dx
        self.xmin = self.mesh.xmin
        self.xmax = self.mesh.xmax
        self.dom_len = self.mesh.dom_len
        self.nn = self.mesh.nn
        self.nelem = self.mesh.nelem
        self.nen = self.mesh.nen
        self.qshape = (self.nen*self.neq_node,self.nelem)

        self.a = self.afun(self.x_elem)
        self.ader = self.afunder(self.x_elem)
        if np.min(self.a) <= 0.: print('WARNING: Variable coefficient a(x) should be >=0')
        












