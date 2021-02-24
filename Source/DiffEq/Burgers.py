#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 12:41:26 2020

@author: andremarchildon
"""

import numpy as np

from Source.DiffEq.DiffEqBase import PdeBaseCons
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
    pde_order = 1

    def __init__(self, para=None, obj_name=None, q0_type='SinWave',
                 use_split_form=False):

        super().__init__(para, obj_name, q0_type)
        self.use_split_form = use_split_form

    def calcE(self, q):

        E = 0.5*q**2
        return E

    def dEdx(self, q):

        if self.use_split_form:
            #q_diag = fn.diag(q)
            #dEdx = (1/3)*fn.gm_gv((fn.lm_gm(self.der1,q_diag) + fn.gm_lm(q_diag,self.der1)), q)
            dEdx = (1/3)*((self.der1 @ q**2) + (q * (self.der1 @ q)))
        else:
            E = self.calcE(q)
            dEdx = self.der1 @ E

        return dEdx

    def dEdq(self, q):

        dEdq = fn.diag(q)
        return dEdq

    def dfdq(self, q):
        # take dEdx as a vector a_i(q) and find matrix d(a_i)/d(q_j)

        if self.use_split_form:
            # these both do the same, but the second is a bit faster
            #dfdq = -(1/3)*(2*fn.lm_gm(self.der1,fn.diag(q)) + fn.diag(self.der1@q) + fn.gm_lm(fn.diag(q),self.der1))
            dfdq = -(1/3)*(2*np.multiply(self.der1[:,:,None],q) + fn.diag(self.der1 @ q) + fn.gm_lm(fn.diag(q),self.der1))
        else:
            # this does the same as the base function, just a bit faster
            dfdq = - np.multiply(self.der1[:,:,None],q)
            
        return dfdq

    def calc_obj(self, *argv):
        return None

    def calc_LF_const(self):
        ''' Constant for the Lax-Friedrichs flux'''
        q = fn.check_q_shape(self.set_q0())
        return np.max(np.abs(q))

    def dEdq_eig_abs(self, dEdq):

        dEdq_eig_abs = abs(dEdq)
        return dEdq_eig_abs


    def sat_der1_ec(self, q_fL, q_fR, sigma=1, avg='simple'):
        '''
        Purpose
        ----------
        Calculate the entropy conservative SAT for Burgers equation
        Parameters
        ----------
        q_fL : np array, shape (neq_node,nelem)
            The extrapolated solution of the left element(s) to the facet(s).
        q_fR : np array, shape (neq_node,nelem)
            The extrapolated solution of the left element(s) to the facet(s).
        sigma: float (default=1)
            if sigma=1, creates upwinding SAT
            if sigma=0, creates symmetric SAT
        Returns
        -------
        satL : np array
            The contribution of the SAT for the first derivative to the element(s)
            on the left.
        satR : np array
            The contribution of the SAT for the first derivative to the element(s)
            on the right.
        '''
        qLqR = q_fL * q_fR
        qL2 = q_fL**2
        qR2 = q_fR**2

        satL = self.rrR @ (qL2/3 - qLqR/6 - qR2/6)    # SAT for the left of the interface
        satR = self.rrL @ (-qR2/3 + qLqR/6 + qL2/6)    # SAT for the right of the interface

        return satL, satR