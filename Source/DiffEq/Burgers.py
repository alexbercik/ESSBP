#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 12:41:26 2020

@author: bercik
"""

import numpy as np

from Source.DiffEq.DiffEqBase import PdeBase
import Source.Methods.Functions as fn
from numba import njit
from scipy.optimize import root_scalar

class Burgers(PdeBase):
    '''
    Purpose
    ----------
    This class provides the required functions to solve the Burgers equation
    '''

    # Diffeq info
    diffeq_name = 'Burgers'
    dim = 1
    neq_node = 1    # 1 equation in 1D
    eq_type = 'pde'
    pde_order = 1
    has_exa_sol = True

    def __init__(self, para=None, q0_type='SinWave',
                 use_split_form=False, split_alpha=2/3):

        super().__init__(para, q0_type)
        self.use_split_form = use_split_form
        self.split_alpha = split_alpha

    def exact_sol(self, time=0, x=None):

        if x is None:
            x = self.x_elem

        reshape = False
        if x.ndim >1: 
            reshape=True
            x = x.flatten('F')

        #u0 = self.set_q0(xy=x)
        #a,b = np.min(u0), np.max(u0) # u(x,t) must be some u0(x), so bounded by min & max
        #a,b = a-(b-a)/20 , b+(b-a)/20 # expand the boundaries slightly just in case
        u = np.empty_like(x) # initiate u
        for i in range(len(x)):
            # the below is more efficient, but I couldn't figure out periodic boundaries
            #f = lambda z : self.set_q0(xy=z) # f(x) = u0(x+u0*t0) = u0(x)
            #eq = lambda z : f(x[i]-z*time) - z  # find roots u of 0 = f(x-ut) - u
            #u[i] = bisect(eq,a,b,xtol=1e-12,maxiter=1000)

            u0 = lambda x0 : self.set_q0(xy=x0)
            eq = lambda x0 : np.mod((x[i] - self.xmin) - u0(x0)*time, self.dom_len) + self.xmin - x0
            #eq = lambda x0 : x[i] - u0(x0)*time - x0
            res = root_scalar(eq,bracket=[self.xmin-0.01,self.xmax+0.01],method='secant',
                             x0=x[i]-u0(x[i])*time,xtol=1e-12,maxiter=1000)
            x0 = res.root
            u[i] = u0(x0)
        
        if reshape:
            u = np.reshape(u,(self.nen,self.nelem),'F')

        return u

    def calcEx(self, q):
        E = 0.5*q**2
        return E

    def dExdx(self, q, E):

        if self.use_split_form:
            dExdx = (self.split_alpha/2.) * self.gm_gv(self.Dx, q**2) \
                  + (1.-self.split_alpha) * q * self.gm_gv(self.Dx, q)
        else:
            dExdx = self.gm_gv(self.Dx, E)

        return dExdx

    def dExdq(self, q):

        dExdq = fn.gdiag_to_gbdiag(q)
        return dExdq
    
    def d2Exdq2(self, q):
        
        d2Exdq2 = fn.gdiag_to_gm(np.ones(q.shape))
        return d2Exdq2

    def dfdq(self, q):
        # take dExdx as a vector a_i(q) and find matrix d(a_i)/d(q_j)

        if self.use_split_form:
            # these both do the same, but the second is a bit faster
            #dfdq = -(1/3)*(2*fn.lm_gm(self.der1,fn.gdiag_to_gm(q)) + fn.gdiag_to_gm(self.der1@q) + fn.gm_lm(fn.gdiag_to_gm(q),self.der1))
            #dfdq = -(1/3)*(2*np.multiply(self.der1[:,:,None],q) + fn.gdiag_to_gm(self.der1 @ q) + fn.gm_lm(fn.gdiag_to_gm(q),self.der1))
            dfdq = -self.split_alpha*fn.gm_gv_colmultiply(self.Dx,q) - (1-self.split_alpha)*(fn.gdiag_to_gm(fn.gm_gv(self.Dx, q)) + fn.gdiag_gm(q,self.Dx))
        else:
            # this does the same as the base function, just a bit faster
            dfdq = - fn.gm_gv_colmultiply(self.Dx,q)
            
        return dfdq

    def calc_LF_const(self):
        ''' Constant for the Lax-Friedrichs flux. Not needed for SBP.'''
        q = fn.check_q_shape(self.set_q0())
        return np.max(np.abs(q))

    def dExdq_abs(self, q, entropy_fix):

        dExdq_abs = fn.gdiag_to_gbdiag(abs(q))
        return dExdq_abs
    
    def maxeig_dExdq(self, q):
        ''' return the absolute maximum eigenvalue - used for LF fluxes '''
        return np.abs(q)
    
    def maxeig_dEndq(self, q, dxidx):
        ''' return the absolute maximum eigenvalue - used for LF fluxes '''
        return np.abs(q*dxidx)
    
    def entropy_var(self,q):
        return q
    
    def dqdw(self,q):
        return fn.gdiag_to_gm(np.ones(q.shape))
    
    def maxeig_dqdw(self,q):
        return np.ones(q.shape)
    
    def dExdw_abs(self, q, entropy_fix):

        dExdw_abs = fn.gdiag_to_gbdiag(abs(q))
        return dExdw_abs
    
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
    
    def calc_breaking_time(self,sig_fig=3):
        ''' estimate the time at which the solution breaks '''
        q0 = self.set_q0()
        dqdx = self.gm_gv(self.Dx, q0)
        Tb = -1/np.min(dqdx)
        print(f'The breaking time is approximately T = {Tb:.{sig_fig}g}')
    
    @njit
    def ec_flux(qL,qR):
        ''' entropy conservative 2-point flux for the Hadamard form '''
        fx = (qL**2 + qL*qR + qR**2)/6
        return fx
    
    @njit   
    def central_flux(qL,qR):
        ''' a central 2-point flux for hadamard form.
        This allows us to jit the hadamard flux functions. '''
        f = fn.arith_mean(qL**2/2,qR**2/2)
        return f

