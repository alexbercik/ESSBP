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
    npar = 0        # No. of design parameters
    pde_order = 1 # Order of the highest derivative in the PDE
    xy = None
    has_exa_sol = True
    para_names = ('ax','ay','az')

    def __init__(self, para=None, obj_name=None, q0_type='SinWave'):

        super().__init__(para, obj_name, q0_type)
        self.ax = self.para[0]
        self.ay = self.para[1]
        self.az = self.para[2]

    def exact_sol(self, time=0):

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
        
        #dEdq = np.array(self.a,ndmin=(q.ndim+1)) DO NOT USE!
        dEdq = fn.diag(np.ones(q.shape)*self.ax)
        return dEdq
    
    def dEydq(self, q):

        dFdq = fn.diag(np.ones(q.shape)*self.ay)
        return dFdq
    
    def dEzdq(self, q):

        dGdq = fn.diag(np.ones(q.shape)*self.az)
        return dGdq

    def dExdq_eig_abs(self, dExdq):

        dEdq_eig_abs = np.abs(dExdq)
        return dEdq_eig_abs
    
    def dEydq_eig_abs(self, dEydq):

        dEdq_eig_abs = np.abs(dEydq)
        return dEdq_eig_abs
    
    def dEzdq_eig_abs(self, dEzdq):

        dEdq_eig_abs = np.abs(dEzdq)
        return dEdq_eig_abs
    
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

        dEdq = fn.diag(np.zeros(q.shape))
        return dEdq
    
    def d2Eydq2(self, q):

        dEdq = fn.diag(np.zeros(q.shape))
        return dEdq

    def d2Ezdq2(self, q):

        dEdq = fn.diag(np.zeros(q.shape))
        return dEdq
    
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
    
    @njit   
    def central_fix_Ez(qL,qR):
        ''' a central 2-point flux for hadamard form but with az fixed at 1.
        This allows us to jit the hadamard flux functions. '''
        f = fn.arith_mean(qL,qR)
        return f