#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 13:16:51 2020

@author: andremarchildon
"""

import numpy as np

from Source.DiffEq.DiffEqBase import PdeBaseCons
from Source.DiffEq.SatBase import SatBaseCons
from Source.DiffEq.NumFlux import NumFlux
import Source.Methods.Functions as fn


class LinearConv(PdeBaseCons):
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
    xy = None
    has_exa_sol = True
    para_names = ('a',)

    def __init__(self, para=None, obj_name=None, q0_type='SinWave'):

        super().__init__(para, obj_name, q0_type)
        self.a = self.para[0]

    def exact_sol(self, time=0):

        assert self.dim==1, 'exact sol only setup for 1D'

        if self.xy.ndim == 1:
            xx = self.xy
        else:
            xx = self.xy[:,0]

        xy_mod = np.mod((xx - self.xmin) - self.a*time, self.dom_len) + self.xmin
        exa_sol = self.set_q0(xy=xy_mod)

        return exa_sol

    def dfdq(self, q, xy_idx0=None, xy_idx1=None):

        dfdq = self.diag(np.zeros(q.shape))
        #dfdq = 0
        return dfdq

    def calcE(self, q):

        E = self.a * q
        return E

    def dEdq(self, q):
        
        #dEdq = np.array(self.a,ndmin=(q.ndim+1)) DO NOT USE!
        # Causes error in fn.gm_gv method in sat_der1_upwind when structured
        dEdq = fn.diag(np.ones(q.shape)*self.a)
        return dEdq

class LinearConvFd(LinearConv):
    ''' Solve the linear convection eq with finite difference '''

class LinearConvSbp(SatBaseCons, LinearConv):
    ''' Solve the linear convection eq with SBP operators '''

    def dfdq_sat_der1_upwind(self, q_fA, q_fB):
        return self.dfdq_sat_der1_upwind_scalar(q_fA, q_fB)

    def dEdq_eig_abs(self, dEdq):

        dEdq_eig_abs = np.abs(dEdq)
        return dEdq_eig_abs

    def d2Edq2(self, q):
        n = q.size
        return np.zeros((n,n))

class LinearConvDg(LinearConv, NumFlux):
    ''' Solve the linear convection eq with DG '''

    def calc_LF_const(self):
        ''' Constant for the Lax-Friedrichs flux'''
        return abs(self.a)
