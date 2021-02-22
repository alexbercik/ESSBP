#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 15:28:49 2020

@author: bercik
"""
import numpy as np

'''
This file has one class: NumFlux. This determines the numerical flux used by 
the DG scheme. So far these are set up strictly for dim=1.
TODO: Look carefully at how the averaging is being done. Average E? or q?
'''

class NumFlux:
    
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
            self.numflux = lambda qA,qB: self.LF(qA,qB,0,0)
        elif method == 'upwind' or method == 'upwind_global' or method == 'LF_upwind_global':
            C = self.calc_LF_const()
            # TODO: global and local need to be defined better as I don't pass in q...
            self.numflux = lambda qA,qB: self.LF_scalar(qA,qB,1,C)
        elif method == 'split':
            if self.diffeq_name == 'Burgers':
                self.numflux = lambda qA,qB: self.split(qA,qB,2/3)
            else:
                print('ERROR: split flux not set up for this Diffeq')
        elif method == 'tadmor':
            if self.diffeq_name == 'Burgers':
                self.numflux = lambda qA,qB: self.tadmor(qA,qB,2/3)
        elif method == 'rusanov':
            if self.neq_node == 1:
                self.numflux = self.rusanov_scalar
        elif method == 'rusanov_gassner':
            if self.neq_node == 1:
                self.numflux = self.rusanov_gassner_scalar
            else:
                print('ERROR: tadmor flux not set up for this Diffeq')
        else:
            raise Exception('Choice of Numerical Flux not understood.')
            
    def LF(self, qA, qB, alpha, C):
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
        avgE = (self.calcE(qA) + self.calcE(qB))/2
        # TODO: Do I want different kinds of averaging here? Then no longer LF...
        flux = avgE + 0.5*C*alpha*(qA-qB)
        return flux, -flux
    
    def split(self, qA, qB, alpha, avg='simple_E'):
        '''
        A split form skew-symmetric flux. Yields EC flux with alpha=2/3 for
        Burgers equation and alpha=1/2 for variable coeff. advection.
        
        Need to modify diffq term for general eq. Only works for Burgers. (See Gassner et all eq 17)

        Parameters
        ----------
        qA : np array
            The solution at the flux node in the element to the left of the interface.
        qB : np array
           The solution at the flux node in the element to the right of the interface.
        alpha : float
            constant used to determine split form. use alpha=2/3 for Burgers.
        '''
        assert(self.diffeq_name == 'Burgers'),'Not set up for other equation types'
        if avg=='simple':
            avgE = self.calcE((qA + qB)/2)
        elif avg=='simple_E':
            avgE = (self.calcE(qA) + self.calcE(qB))/2
        elif avg=='roe':
            raise Exception('Roe Average not coded up yet')
        else:
            raise Exception('Averaging method not understood.')
            
        diffq = qB - qA
        flux = avgE - (1-alpha)*(diffq**2)/4
        return flux, -flux
    
    def tadmor(self, qA, qB, alpha):
        '''
        A split form skew-symmetric flux from Tadmor 2003 that attempts to
        remove anti-dissipative components. Yields modified EC flux with 
        alpha=2/3 for Burgers equation.
        
        Need to modify diffq term for general eq. Only works for Burgers. (See Gassner et all eq 23)

        Parameters
        ----------
        qA : np array
            The solution at the flux node in the element to the left of the interface.
        qB : np array
           The solution at the flux node in the element to the right of the interface.
        alpha : float
            constant used to determine split form. use alpha=2/3 for Burgers.
        '''
        assert(self.diffeq_name == 'Burgers'),'Not set up for other equation types'
        avgE = (self.calcE(qA) + self.calcE(qB))/2
        diffq = qB - qA
        coeff = np.maximum((1-alpha)*diffq/2,0)
        flux = avgE - coeff*diffq/2
        return flux, -flux
    
    def rusanov_scalar(self, qA, qB):
        '''
        The standard Rusanov flux function, see Gassner 2020 or Toro, E.F. 2009
        '''
        assert(self.diffeq_name == 'Burgers'),'Not set up for other equation types'
        avgE = (self.calcE(qA) + self.calcE(qB))/2
        diffq = qB - qA
        coeff = np.maximum(abs(qA),abs(qB))
        flux = avgE - coeff*diffq/2
        return flux, -flux
    
    def rusanov_gassner_scalar(self, qA, qB):
        '''
        The modified Rusanov flux function from Gassner 2020 (stability issues)
        This should probably only be used for Burgers equation
        '''
        assert(self.diffeq_name == 'Burgers'),'Not set up for other equation types'
        avgE = (self.calcE(qA) + self.calcE(qB))/2
        diffq = qB - qA
        coeff = diffq/6 + np.maximum(abs(qA),abs(qB))
        flux = avgE - coeff*diffq/2
        return flux, -flux 