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
    dim = 3
    neq_node = 1    # 1 equation in 1D
    pde_order1 = True
    pde_order2 = False
    xy = None
    has_exa_sol = True
    para_names = ('ax','ay','az')
    ax_fix = 1
    ay_fix = 1
    az_fix = 1
    para_fix = [ax_fix,ay_fix,az_fix]

    def __init__(self, para=None, q0_type='SinWave'):

        super().__init__(para, q0_type)
        self.ax = self.para[0]
        self.ay = self.para[1]
        self.az = self.para[2]

        if self.ax == self.ax_fix:
            print('Using the fixed ax={} diffeq functions since params match.'.format(self.ax_fix))
            def dExdq_fix(q):
                nen,nelem = np.shape(q)
                return np.ones((nen,1,1,nelem),dtype=q.dtype)
            self.dExdq = dExdq_fix
            self.dExdq_abs = dExdq_fix
            self.maxeig_dExdq = lambda q : np.ones(q.shape)
            self.central_Ex = self.central_fix_Ex
        
        if self.ay == self.ay_fix:
            print('Using the fixed ay={} diffeq functions since params match.'.format(self.ay_fix))
            def dEydq_fix(q):
                nen,nelem = np.shape(q)
                return np.ones((nen,1,1,nelem),dtype=q.dtype)
            self.dEydq = dEydq_fix
            self.dEydq_abs = dEydq_fix
            self.maxeig_dEydq = lambda q : np.ones(q.shape)
            self.central_Ey = self.central_fix_Ey

        if self.az == self.az_fix:
            print('Using the fixed ay={} diffeq functions since params match.'.format(self.az_fix))
            def dEzdq_fix(q):
                nen,nelem = np.shape(q)
                return np.ones((nen,1,1,nelem),dtype=q.dtype)
            self.dEzdq = dEzdq_fix
            self.dEzdq_abs = dEzdq_fix
            self.maxeig_dEzdq = lambda q : np.ones(q.shape)
            self.central_Ez = self.central_fix_Ez

    def exact_sol(self, time=0, xyx=None, guess=None):

        if xyz is None:
            xyz = self.xyz_elem
        xyz_mod = np.empty(xyz.shape)

        xyz_mod[:,0,:] = np.mod((xyz[:,0,:] - self.xmin[0]) - self.ax*time, self.dom_len[0]) + self.xmin[0]
        xyz_mod[:,1,:] = np.mod((xyz[:,1,:] - self.xmin[1]) - self.ay*time, self.dom_len[1]) + self.xmin[1]
        xyz_mod[:,2,:] = np.mod((xyz[:,2,:] - self.xmin[2]) - self.az*time, self.dom_len[2]) + self.xmin[2]
        exa_sol = self.set_q0(xy=xyz_mod)

        return exa_sol

    def calcEx(self, q):

        E = self.ax * q
        return E
    
    def calcEy(self, q):

        F = self.ay * q
        return F
    
    def calcEz(self, q):

        G = self.az * q
        return G

    def dExdq(self, q):
        nen,nelem = np.shape(q)
        dExdq = self.ax*np.ones((nen,1,1,nelem),dtype=q.dtype)
        return dExdq
    
    def dEydq(self, q):
        nen,nelem = np.shape(q)
        dEydq = self.ay*np.ones((nen,1,1,nelem),dtype=q.dtype)
        return dEydq
    
    def dEzdq(self, q):
        nen,nelem = np.shape(q)
        dEzdq = self.az*np.ones((nen,1,1,nelem),dtype=q.dtype)
        return dEzdq

    def dExdq_abs(self, q, entropy_fix):
        nen,nelem = np.shape(q)
        dExdq = abs(self.ax)*np.ones((nen,1,1,nelem),dtype=q.dtype)
        return dExdq
    
    def dEydq_abs(self, q, entropy_fix):
        nen,nelem = np.shape(q)
        dEydq = abs(self.ay)*np.ones((nen,1,1,nelem),dtype=q.dtype)
        return dEydq
    
    def dEzdq_abs(self, q, entropy_fix):
        nen,nelem = np.shape(q)
        dEzdq = abs(self.az)*np.ones((nen,1,1,nelem),dtype=q.dtype)
        return dEzdq
    
    def maxeig_dExdq(self, q):
        ''' return the maximum eigenvalue - used for LF fluxes '''
        maxeig = np.ones(q.shape)*self.ax
        return maxeig

    def maxeig_dEydq(self, q):
        ''' return the maximum eigenvalue - used for LF fluxes '''
        maxeig = np.ones(q.shape)*self.ay
        return maxeig
    
    def maxeig_dEzdq(self, q):
        ''' return the maximum eigenvalue - used for LF fluxes '''
        maxeig = np.ones(q.shape)*self.az
        return maxeig
    
    def d2Exdq2(self, q):
        nen,nelem = np.shape(q)
        dExdq = np.zeros((nen,1,1,nelem),dtype=q.dtype)
        return dExdq
    
    def d2Eydq2(self, q):
        nen,nelem = np.shape(q)
        dEydq = np.zeros((nen,1,1,nelem),dtype=q.dtype)
        return dEydq

    def d2Ezdq2(self, q):
        nen,nelem = np.shape(q)
        dEzdq = np.zeros((nen,1,1,nelem),dtype=q.dtype)
        return dEzdq
    
    @njit   
    def central_fix_fluxes(qL,qR):
        ''' a central 2-point flux for hadamard form but with ax fixed at (1,1,1).
        This allows us to jit the hadamard flux functions. '''
        f = fn.arith_mean(qL,qR)
        return f, f, f