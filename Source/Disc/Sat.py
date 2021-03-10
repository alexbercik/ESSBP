#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 18:42:28 2021

@author: bercik
"""
#import numpy as np

from Source.Disc.SatDer1 import SatDer1
from Source.Disc.SatDer2 import SatDer2


class Sat(SatDer1, SatDer2):

    # TODO: Finish the comments here
    '''
    The methods calc_sat_unstruc,
    '''

    def init_sat_class(self, method):
        
        #TODO: Not complete, only tested for scalar PDE

        assert self.dim == 1, 'SATs only for 1D are available'

        ''' Set the methods that will be used to calculate the SATs '''

        if self.pde_order == 1:
            
            if method == 'central':
                if self.neq_node == 1:
                    self.calc_sat = lambda qL,qR: self.sat_der1_upwind(qL, qR, 0)
                    self.calc_dfdq_sat = lambda qL,qR: self.dfdq_sat_der1_upwind_scalar(qL, qR, 0)
                else:
                    self.calc_sat = lambda qL,qR: self.sat_der1_upwind(qL, qR, 0) # use Roe average?
                    self.calc_dfdq_sat = self.dfdq_sat_der1_complexstep
            elif method == 'upwind':
                if self.neq_node == 1:
                    self.calc_sat = lambda qL,qR: self.sat_der1_upwind(qL, qR, 1) 
                    self.calc_dfdq_sat = lambda qL,qR: self.self.dfdq_sat_der1_upwind_scalar(qL, qR, 1)
                else:
                    self.calc_sat = lambda qL,qR: self.sat_der1_upwind(qL, qR, 1) # use Roe average?
                    self.calc_dfdq_sat = self.dfdq_sat_der1_complexstep
            elif (method.lower()=='ec' and self.diffeq.diffeq_name=='Burgers') or method.lower()=='burgers ec':
                    self.calc_sat = self.sat_der1_burgers_ec
                    self.calc_dfdq_sat = self.dfdq_sat_der1_burgers_ec
            elif (method.lower()=='ec' and self.diffeq.diffeq_name=='Quasi1dEuler') or method.lower()=='crean ec':
                    self.calc_sat = self.sat_der1_crean_ec
                    #self.calc_dfdq_sat = complex step?
            elif (method.lower()=='es' and self.diffeq.diffeq_name=='Quasi1dEuler') or method.lower()=='crean es':
                    self.calc_sat = self.sat_der1_crean_es
                    #self.calc_dfdq_sat = complex step?
            # TODO: Add 'try' if it is there, if not revert to complexstep
            else:
                raise Exception('Choice of SAT not understood.')

            # TODO: Use hasattribute to check for additional parameters?
            # Set the method for the sat and dfdq_sat for the first derivative
            #self.calc_sat_der1 = getattr(self, self.diffeq.sat_type_der1)
            #self.calc_dfdq_sat_der1 = getattr(self, self.diffeq.dfdq_sat_type_der1)

        elif self.pde_order == 2:
            
            # TODO

            # Set the method for the sat and dfdq_sat for the various derivatives
            self.sat_der1 = getattr(self, self.diffeq.sat_type_der1)
            self.sat_der2 = getattr(self, self.diffeq.sat_type_der2)

            self.dfdq_sat_der1 = getattr(self, self.diffeq.dfdq_sat_type_der1)
            self.dfdq_sat_der2 = getattr(self, self.diffeq.dfdq_sat_type_der2)

            self.calc_sat = self.sat_der2_master
            self.calc_dfdq_sat = self.dfdq_sat_der2_master

        else:
            raise Exception('SAT methods for reqested order of PDE is not available')
            
  
    def get_interface_sol(self, qelemL, qelemR):
        '''
        Parameters
        ----------
        qelemL/R : np array
            The solutions at all the nodes in one (or many) element(s) either
            on the left or right of the interface.

        Returns
        -------
        q_f : np array
            Solution at the desired facet(s).
        '''

        # If qelem is of size self.neq_node, then q is evaluated at the facet
        # already. This can be the case for boundary interfaces or when using
        # the complex step method
        if qelemL.size == self.neq_node:
            q_fL = qelemL
        else:
            q_fL = self.rrR.T @ qelemL
            
        if qelemR.size == self.neq_node:
            q_fR = qelemR
        else:
            q_fR = self.rrL.T @ qelemR

        return q_fL , q_fR