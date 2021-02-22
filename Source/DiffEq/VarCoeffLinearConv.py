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


class VarCoeffLinearConv(PdeBaseCons):
    '''
    Purpose
    ----------
    This class provides the required functions to solve the variable coefficient
    linear convection equation (in conservative form, u_t + (a(x)u)_x = 0 ):
    '''

    diffeq_name = 'VariableCoefficientLinearConvection'
    dim = 1
    neq_node = 1    # 1 equation in 1D
    npar = 0        # No. of design parameters
    xy = None
    has_exa_sol = False
    para_names = ('a',)

    def __init__(self, para=None, obj_name=None, q0_type='SinWave'):

        super().__init__(para, obj_name, q0_type)
        if self.para is None:
            print('Setting initial condition as a(x) coefficient')
            self.para = self.q0_type
        else:
            assert(isinstance(self.para, str)),'Set self.para as None or q0_type string'
            
    def set_mesh(self, mesh):
        super(VarCoeffLinearConv, self).set_mesh(mesh)
        self.a = lambda x: self.set_q0(q0_type=self.para, xy=x)

    def dfdq(self, q, xy_idx0=None, xy_idx1=None):
        raise Exception('Not coded up yet')

    def calcE(self, q, xy):

        E = self.a(xy) * q
        return E

    def dEdq(self, q, xy):

        dEdq = fn.diag(self.a(xy))
        return dEdq
    
    def dEdx(self, q, xy):        
        a = self.a(xy)      
        dEdx = self.alpha*(self.der1 @ (a*q)) + (1-self.alpha)*(a*(self.der1 @ q) + q*(self.der1 @ a))
        return dEdx
    
    def dqdt(self, q, xy):

        dEdx = self.dEdx(q, xy)

        dqdt = -dEdx
        return dqdt
    
    def calc_LF_const(self,q=None,use_local=False,xy=None):
        ''' Calculate the constant for the Lax-Friedrichs flux, also useful
        to set the CFL number. Equal to max(dEdq). See Hesthaven pg. 33. 
        If q is given, use that. If not, use the initial condition.
        If use_local=True, uses a local lax-friedrichs flux constant, which is 
        less dissipative, and returns a vector of shape (1,nelem). Otherwise 
        it uses a global lax-friedrichs flux constant, and returns a float.'''
        if q==None:
            q = self.set_q0()
        if self.neq_node == 1: # scalar
            if use_local:
                return np.max(np.abs(self.dEdq(q,xy)),axis=(0,1))
            else:
                return np.max(np.abs(self.dEdq(q,xy)))
        else: # system
            # TODO: Could save a variable containing dEdq, eigenvalues, eigenvectors
            # so that I don't have to compute it multiple times for different
            # parts of the code
            dEdq_mod = np.transpose(self.dEdq(q,xy),axes=(2,0,1))
            eig_val = np.linalg.eigvals(dEdq_mod)
            if use_local:
                return np.max(np.abs(eig_val),axis=1)
            else:
                return np.max(np.abs(eig_val))

class LinearConvFd(VarCoeffLinearConv):
    ''' Solve the linear convection eq with finite difference '''

class LinearConvSbp(SatBaseCons, VarCoeffLinearConv):
    ''' Solve the linear convection eq with SBP operators '''

    def dEdq_eig_abs(self, dEdq):

        dEdq_eig_abs = np.abs(dEdq)
        return dEdq_eig_abs

    def d2Edq2(self, q):
        n = q.size
        return np.zeros((n,n))
    
    def get_interface_sol(self, qelem, xy, is_left_of_facet):
        '''
        Parameters
        ----------
        qelem : np array
            The solution at all the nodes in one element.
        is_left_of_facet : bool
            True if the solution is to be extrapolated to the left facet,
            false for the right facet

        Returns
        -------
        q_f : np array
            Solution at the desired facet.
        '''

        # If qelem is of size self.neq_node, then q is evaluated at the facet
        # already. This can be the case for boundary interfaces or when using
        # the complex step method
        if qelem.size == self.neq_node:
            q_f = qelem
        else:
            if is_left_of_facet:
                q_f = self.rrR.T @ qelem
                x_f = self.rrR.T @ xy
            else:
                q_f = self.rrL.T @ qelem
                x_f = self.rrL.T @ xy

        return q_f , x_f

    def sat_der1_upwind(self, q_fA, q_fB, xy_A, xy_B, sigma, avg='simple'):
        '''
        Purpose
        ----------
        Calculate the upwind SAT for a first derivative term such as
        \frac{dE}{dx}) where E can be nonlinear

        Parameters
        ----------
        q_fA : np array
            The solution to the left of the facet.
        q_fB : np array
            The solution to the right of the facet.
        sigma: float
            if sigma=1, creates upwinding SAT
            if sigma=0, creates symmetric SAT

        Returns
        -------
        satA : np array
            The contribution of the SAT for the first derivative to the element
            on the left.
        satB : np array
            The contribution of the SAT for the first derivative to the element
            on the right.
        '''
        
        if avg=='simple':
            qfacet = (q_fA + q_fB)/2 # Alternatively, a Roe average can be used
        elif avg=='roe':
            raise Exception('Roe Average not coded up yet')
        else:
            raise Exception('Averaging method not understood.')
            
        x_facet = (xy_A + xy_B)/2
        
        A = self.dEdq(qfacet, x_facet)
        A_abs = self.dEdq_eig_abs(A)

        # Upwinding flux
        A_upwind = (A + sigma*A_abs)/2
        A_downwind = (A - sigma*A_abs)/2

        # Calculate the correction
        k = q_fA - q_fB
        # !!!!!!!! This part might be wrong.....

        # SAT for the left of the interface (SAT_N)
        satA = self.rrR @ fn.gm_gv(A_downwind, k)  # SAT for the left of the interface
        satB = self.rrL @ fn.gm_gv(A_upwind, k)    # SAT for the right of the interface

        return satA, satB
    
    def set_sat(self, method):
        '''
        Purpose
        ----------
        Set the method used to calculate the numerical flux.

        Parameters
        ----------
        method : str
            The desired method.
        '''
        #TODO: Not complete, only tested for scalar PDE
        if method == 'central':
             self.sat_der1 = lambda qA,qB,xA,xB: self.sat_der1_upwind(qA, qB, 0, xA, xB)
             self.dfdq_sat_der1 = lambda qA,qB,xA,xB: self.dfdq_sat_der1_upwind_scalar(qA, qB, 0, xA,xB)
        elif method == 'upwind':
            self.sat_der1 = lambda qA,qB,xA,xB: self.sat_der1_upwind(qA, qB, 1, xA,xB) 
            self.dfdq_sat_der1 = lambda qA,qB,xA,xB: self.self.dfdq_sat_der1_upwind_scalar(qA, qB, 1, xA,xB)
        else:
            raise Exception('Choice of SAT not understood.')

    def calc_sat(self, qelemA, qelemB, xA, xB):
        '''
        Purpose
        ----------
        Default function to calculate the upwind SATs for a PDE with only
        first derivatives.

        Parameters
        ----------
        qelemA : np array
            The solution at the nodes in the element to the left of the interface.
        qelemB : np array
           The solution at the nodes in the element to the right of the interface.

        Returns
        -------
        The SAT contribution to the elements on both sides of the interface.
        '''

        q_fA, x_fA = self.get_interface_sol(qelemA, xA, is_left_of_facet=True)
        q_fB, x_fB = self.get_interface_sol(qelemB, xB, is_left_of_facet=False)

        return self.sat_der1(q_fA, q_fB, x_fA, x_fB)

class LinearConvDg(VarCoeffLinearConv, NumFlux):
    ''' Solve the linear convection eq with DG '''

    def set_numflux(self, method):
        '''
        Purpose
        ----------
        Set the method used to calculate the numerical flux.

        Parameters
        ----------
        method : str
            The desired method.
        '''

        if method == 'central':
            self.numflux = lambda qA,qB,xA,xB: self.LF(qA,qB,xA,xB,0,0)
        elif method == 'upwind' or method == 'upwind_global' or method == 'LF_upwind_global':
            C = self.calc_LF_const()
            # TODO: global and local need to be defined better as I don't pass in q...
            self.numflux = lambda qA,qB,xA,xB: self.LF(qA,qB,xA,xB,1,C)
        else:
            raise Exception('Choice of Numerical Flux not understood.')
            
    def LF(self, qA, qB,xA,xB, alpha, C):
        '''
        Lax-Friedrichs flux (works for scalar and 1D system).

        Parameters
        ----------
        qA : np array
            The solution at the flux node in the element to the left of the interface.
        qB : np array
           The solution at the flux node in the element to the right of the interface.
        alpha : float
            upwinding parameter. alpha=0 yields central flux, alpha=1 yields
            upwind flux
        C : float or np array
            the constant that controls dissipation. This should be greater 
            than max dEdq (either locally or globally).
            
        TODO: generalize for cases where qA and qB are not on the boundary

        Returns
        -------
        The numerical flux on both sides of the interface.
        '''
        avgE = (self.calcE(qA,xA) + self.calcE(qB,xB))/2
        # TODO: Do I want different kinds of averaging here? Then no longer LF...
        flux = avgE + 0.5*C*alpha*(qA-qB)
        return flux, -flux
    
