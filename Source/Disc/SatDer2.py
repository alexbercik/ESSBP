#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 19:24:28 2021

@author: bercik
"""

#import numpy as np
#import Source.Methods.Functions as fn


class SatDer2:
    
    def get_interface_sol_der2(self, qL, qR):

        ''' Get the solution to the left of the facet '''
        if qL.size == self.neq_node:
            at_bdy_L = True
            q_fL = self.diffeq.qL
            qder1_fL = self.diffeq.q_derL
        else:
            at_bdy_L = False
            q_fL = self.tRT @ qL
            qder1_fL = self.tRT @ self.der1 @ qL

        ''' Get the solution to the right of the facet '''
        if qR.size == self.neq_node:
            at_bdy_R = True
            q_fR = self.diffeq.qR
            qder1_fR = self.diffeq.q_derR
        else:
            at_bdy_R = False
            q_fR = self.tLT @ qL
            qder1_fR = self.tLT @ self.der1 @ qR

        return (q_fL, qder1_fL, at_bdy_L), (q_fR, qder1_fR, at_bdy_R)


    def sat_der2_master(self, qL, qR):
        '''
        Parameters
        ----------
        qL : np array
            The solution at the nodes in the element(s) to the left of the interface.
        qR : np array
            The solution at the nodes in the element(s) to the right of the interface.
        '''

        # Get required terms at the boundaries
        q_fL_all, q_fR_all = self.get_interface_sol_der2(qL, qR)
        q_fL, qder1_fL, at_bdy_L = q_fL_all
        q_fR, qder1_fR, at_bdy_R = q_fR_all

        # Get SATs for each derivative term
        satL_der1, satR_der1 = self.calc_sat_der1(q_fL, q_fR)
        satL_der2, satR_der2 = self.calc_sat_der2(q_fL, q_fR, qder1_fL, qder1_fR)

        # Add all the SAT contributions
        if at_bdy_L:
            # TODO - check andre's code if hes updated this yet
            satL = None
        else:
            satL = satL_der1 + satL_der2

        if at_bdy_R:
            # TODO - check andre's code if hes updated this yet
            satR = None
        else:
            satR = satR_der1 + satR_der2

        return satL, satR

    def dfdq_sat_der2_master(self, qL, qR):
        '''
        Parameters
        ----------
        qL : np array
            The solution at the nodes in the element(s) to the left of the interface.
        qR : np array
            The solution at the nodes in the element(s) to the right of the interface.
        '''
        # Get required terms at the boundaries
        q_fL_all, q_fR_all = self.get_interface_sol_der2(qL, qR)
        q_fL, qder1_fL, at_bdy_L = q_fL_all
        q_fR, qder1_fR, at_bdy_R = q_fR_all

        # Get SATs for each derivative term
        dSatL_dqL_der1, dSatL_dqR_der1, dSatR_dqL_der1, dSatR_dqR_der1 = self.calc_dfdq_sat_der1(q_fL, q_fR)
        dSatL_dqL_der2, dSatL_dqR_der2, dSatR_dqL_der2, dSatR_dqR_der2 = self.calc_dfdq_sat_der2(q_fL, q_fR, qder1_fL, qder1_fR)
        
        # Add all the SAT contributions
        if at_bdy_L:
            # TODO - check andre's code if hes updated this yet
            dSatL_dqL = dSatL_dqR = None
        else:
            dSatL_dqL = dSatL_dqL_der1 + dSatL_dqL_der2
            dSatL_dqR = dSatL_dqR_der1 + dSatL_dqR_der2

        if at_bdy_R:
            # TODO - check andre's code if hes updated this yet
            dSatR_dqR = dSatR_dqL = None
        else:
            dSatR_dqR = dSatR_dqR_der1 + dSatR_dqR_der2
            dSatR_dqL = dSatR_dqL_der1 + dSatR_dqL_der2

        return dSatL_dqL, dSatL_dqR, dSatR_dqL, dSatR_dqR

############################## TODO ##########################################

    def sat_unstruc_der2_scalar(self, q_fA, q_fB, qder1_fA, qder1_fB):
        '''
        Purpose
        ----------
        Calculate the upwind SAT for a linear second derivative scalar term
        (ie \frac{d^2 q}{dx^2}). The SAT weekly enforce:
            u_x at x=0 (sigma1)
            u   at x=1 (sigma2)
        Parameters
        ----------
        q_fA : np array
            The solution to the left of the facet.
        q_fB : np array
            The solution to the right of the facet.
        qder1_fA : np array
            The numerical derivative of the solution at all the nodes in the
            element to the left of the interface.
        qder1_fB : np array
            The numerical derivative of the solution at all the nodes in the
            element to the right of the interface.
        Returns
        -------
        satA : np array
            The contribution of the SAT for the second derivative to the
            element on the left.
        satB : np array
            The contribution of the SAT for the second  derivative to the
            element on the right.
        '''

        sigmaA = self.diffeq.c_der2 /2
        sigmaB = -sigmaA

        satA = sigmaA * (self.der1.T @ self.rrR) * (q_fA - q_fB)
        satB = sigmaB * self.rrL * (qder1_fB - qder1_fA)

        return satA, satB

    def dfdq_sat_unstruc_der2_scalar(self, q_fA, q_fB, qder1_fA, qder1_fB):

        '''
        Derivative of the SAT for a linear scalar 2nd derivative term
        ie \frac{d^2 q}{dx^2}.
        See the function above (sat_der2_scalar_linear).
        '''

        sigmaA = self.diffeq.c_der2 /2
        sigmaB = -sigmaA

        dSatA_dqA = sigmaA * (self.der1.T @ self.rrR @ self.rrR.T)
        dSatA_dqB = -sigmaA * (self.der1.T @ self.rrR @ self.rrL.T)
        dSatB_dqA = sigmaB * (self.rrL @ self.rrR.T @ self.der1)
        dSatB_dqB = -sigmaB * (self.rrL @ self.rrL.T @ self.der1)

        # Old implementation, that is incorect (I think - André)
        # dSatA_dqA = self.c_der2 * (self.der1.T @ self.rrR @ self.rrR.T)
        # dSatA_dqB = -self.c_der2 * (self.der1.T @ self.rrR @ self.rrL.T)
        # dSatB_dqA = self.c_der2 * (self.rrL @ self.rrR.T @ self.der1)
        # dSatB_dqB = -self.c_der2 * (self.rrL @ self.rrL.T @ self.der1)

        return dSatA_dqA, dSatA_dqB, dSatB_dqA, dSatB_dqB