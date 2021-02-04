#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 12:41:26 2020

@author: andremarchildon
"""

import numpy as np

from Source.DiffEq.DiffEqBase import PdeBaseCons
from Source.DiffEq.SatBase import SatBaseCons
from Source.DiffEq.NumFlux import NumFlux
import Source.Methods.Functions as fn

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

    def __init__(self, para=None, obj_name=None, q0_type='SinWave',
                 use_split_form=False):

        super().__init__(para, obj_name, q0_type)
        self.use_split_form = use_split_form

    def calcE(self, q):

        E = 0.5*q**2
        return E

    def dEdx(self, q):

        if self.use_split_form:
            #q_diag = self.diag(q)
            #dEdx = (1/3)*self.m_v((self.lm_m(self.der1,q_diag) + self.m_lm(q_diag,self.der1)), q)
            dEdx = (1/3)*((self.der1 @ q**2) + (q * (self.der1 @ q)))
        else:
            E = self.calcE(q)
            dEdx = self.der1 @ E

        return dEdx

    def dEdq(self, q):

        dEdq = fn.diag(q)
        return dEdq

    def dfdq(self, q):
        # TODO: Does this need a split form? not sure if entirely correct
        print('WARNING: NOT ENTIRELY SURE IF CORRECT.')

        A = self.dEdq(q)
        dGdq = self.dGdq(q)

        dfdq = - fn.lm_gm(self.der1, A) + dGdq
        return dfdq

    def calc_obj(self, *argv):
        return None

    def calc_LF_const(self):
        ''' Constant for the Lax-Friedrichs flux'''
        q = fn.check_q_shape(self.set_q0())
        return np.max(np.abs(q))

class BurgersFd(Burgers):
    ''' Solve the Burgers eq with finite difference operators '''

class BurgersSbp(SatBaseCons, Burgers):
    ''' Solve the Burgers eq with SBP operators '''

    def dEdq_eig_abs(self, dEdq):

        dEdq_eig_abs = abs(dEdq)
        return dEdq_eig_abs
    
class BurgersDg(Burgers, NumFlux):
    ''' Solve the Burgers eq with DG '''
        