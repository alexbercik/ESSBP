#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 19:02:07 2021

@author: bercik
"""
import numpy as np
import Source.Methods.Functions as fn

class SatDer1:

    def sat_der1_master(self, qL, qR):
        '''
        Parameters
        ----------
        qL : np array
            The solution at the nodes in the element(s) to the left of the interface.
        qR : np array
            The solution at the nodes in the element(s) to the right of the interface.
        '''

        q_fL, q_fR = self.get_interface_sol(qL, qR)

        return self.sat_der1(q_fL, q_fR)

    def dfdq_sat_der1_master(self, qL, qR):
        '''
        Parameters
        ----------
        qL : np array
            The solution at the nodes in the element(s) to the left of the interface.
        qR : np array
            The solution at the nodes in the element(s) to the right of the interface.
        '''

        q_fL, q_fR = self.get_interface_sol(qL, qR)

        return self.dfdq_sat_der1(q_fL, q_fR)



    def sat_der1_upwind(self, q_fL, q_fR, sigma=1, avg='simple'):
        '''
        Purpose
        ----------
        Calculate the upwind or central SAT for a first derivative term such as
        \frac{dE}{dx}) where E can be nonlinear
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
        if avg=='simple':
            qfacet = (q_fL + q_fR)/2 # Alternatively, a Roe average can be used
        elif avg=='roe':
            raise Exception('Roe Average not coded up yet')
        else:
            raise Exception('Averaging method not understood.')
            
        qf_diff = q_fL - q_fR

        A = self.diffeq.dEdq(qfacet)
        A_abs = self.diffeq.dEdq_eig_abs(A)

        # Upwinding flux
        A_upwind = (A + sigma*A_abs)/2
        A_downwind = (A - sigma*A_abs)/2

        satL = self.rrR @ fn.gm_gv(A_downwind, qf_diff)  # SAT for the left of the interface
        satR = self.rrL @ fn.gm_gv(A_upwind, qf_diff)    # SAT for the right of the interface

        return satL, satR
    
    def dfdq_sat_der1_upwind_scalar(self, q_fL, q_fR, sigma=1, avg='simple'):
        '''
        Purpose
        ----------
        Calculate the derivative of the upwind SAT for a scalar equation.
        Used for implicit time marching.
        Parameters
        ----------
        q_fL : np array, shape (1,nelem)
            The extrapolated solution of the left element(s) to the facet(s).
        q_fR : np array, shape (1,nelem)
            The extrapolated solution of the left element(s) to the facet(s).
        sigma: float (default=1)
            if sigma=1, creates upwinding SAT
            if sigma=0, creates symmetric SAT
        Returns
        -------
        Given an interface, SatL and SatR are the contributions to either side
        qL are the solutions (NOT extrapolated) on either side. Therefore:
            dSatLdqL : derivative of the SAT in the left element wrt left solution
            dSatLdqR : derivative of the SAT in the left element wrt right solution
            dSatRdqL : derivative of the SAT in the right element wrt left solution
            dSatRdqR : derivative of the SAT in the right element wrt right solution
 
        '''
        if avg=='simple':
            qfacet = (q_fL + q_fR)/2 # Alternatively, a Roe average can be used
            # derivative of qfacet wrt qL and qR
            dqfacetdqL = self.rrR.T / 2
            dqfacetdqR = self.rrL.T / 2
        elif avg=='roe':
            raise Exception('Roe Average not coded up yet')
        else:
            raise Exception('Averaging method not understood.')

        qfacet = (q_fL + q_fR)/2 # Assumes simple averaging, will need to be modified otherwise
        qf_diff = q_fL - q_fR

        # First derivative of the flux wrt q
        A = self.diffeq.dEdq(qfacet)
        #A_abs = self.diffeq.dEdq_eig_abs(A) # actually just absolute value (scalar in 3d format)
        A_abs = abs(A)
        sign_A = np.sign(A)
        
        # second derivative of the flux wrt q
        dAdq = self.diffeq.d2Edq2(qfacet)
        
        # derivative of q_diff wrt qL and qR
        dqf_diffdqL = self.rrR.T
        dqf_diffdqR = - self.rrL.T
        
        factor_qL = fn.gm_lm(dAdq,dqfacetdqL)*qf_diff
        factor_qR = fn.gm_lm(dAdq,dqfacetdqR)*qf_diff
        sigmasignA = sigma*sign_A
        sigmaA_abs = sigma*A_abs
        psigsignA = 1+sigmasignA
        msigsignA = 1-sigmasignA
        ApsigmaA_abs = A + sigmaA_abs
        AmsigmaA_abs = A - sigmaA_abs
        # these do the same thing, but the second is a bit quicker (order of gm_lm needs to be fixed)
        # for derivation of below, see personal notes
        #dSatRdqL = 0.5*fn.lm_gm(self.rrL, factor_qL + fn.gm_lm(A, dqf_diffdqL) + sigma*(sign_A*factor_qL + fn.gm_lm(A_abs, dqf_diffdqL)))
        #dSatRdqR = 0.5*fn.lm_gm(self.rrL, factor_qR + fn.gm_lm(A, dqf_diffdqR) + sigma*(sign_A*factor_qR + fn.gm_lm(A_abs, dqf_diffdqR)))
        #dSatLdqL = 0.5*fn.lm_gm(self.rrR, factor_qL + fn.gm_lm(A, dqf_diffdqL) - sigma*(sign_A*factor_qL + fn.gm_lm(A_abs, dqf_diffdqL)))
        #dSatLdqR = 0.5*fn.lm_gm(self.rrR, factor_qR + fn.gm_lm(A, dqf_diffdqR) - sigma*(sign_A*factor_qR + fn.gm_lm(A_abs, dqf_diffdqR)))       
        dSatRdqL = 0.5*fn.lm_gm(self.rrL,psigsignA*factor_qL + fn.gm_lm(ApsigmaA_abs, dqf_diffdqL))
        dSatRdqR = 0.5*fn.lm_gm(self.rrL,psigsignA*factor_qR + fn.gm_lm(ApsigmaA_abs,dqf_diffdqR))
        dSatLdqL = 0.5*fn.lm_gm(self.rrR,msigsignA*factor_qL + fn.gm_lm(AmsigmaA_abs, dqf_diffdqL))
        dSatLdqR = 0.5*fn.lm_gm(self.rrR,msigsignA*factor_qR + fn.gm_lm(AmsigmaA_abs, dqf_diffdqR))

        return dSatLdqL, dSatLdqR, dSatRdqL, dSatRdqR

    def dfdq_sat_der1_complexstep(self, q_fL, q_fR, eps_imag=1e-30):
        '''
        Purpose
        ----------
        Calculates the derivative of the SATs using complex step. This is
        required for PDEs that are systems and nonlinear, such as the Euler eq.
        This function only works to calculate the derivatives of SATs for
        first derivatives but could be modified to handle higher derivatives.
        More details are given below.
        Parameters
        ----------
        q_fL : np array, shape (neq_node,nelem)
            The extrapolated solution of the left element to the facet.
        q_fR : np array, shape (neq_node,nelem)
            The extrapolated solution of the left element to the facet.
        eps_imag : float, optional
            Size of the complex step
        Returns
        -------
        The derivative of the SAT contribution to the elements on both sides
        of the interface. shapes (nen*neq_node,nen*neq_node,nelem)
        '''

        satL_pert_qL, satR_pert_qL = self.sat_der1(q_fL + eps_imag * 1j, q_fR)
        satL_pert_qR, satR_pert_qR = self.sat_der1(q_fL, q_fR + eps_imag * 1j) 
        
        dSatLdqL = fn.gv_lvT(np.imag(satL_pert_qL) / eps_imag , self.rrR.T)
        dSatLdqR = fn.gv_lvT(np.imag(satL_pert_qR) / eps_imag , self.rrL.T)
        dSatRdqL = fn.gv_lvT(np.imag(satR_pert_qL) / eps_imag , self.rrR.T)
        dSatRdqR = fn.gv_lvT(np.imag(satR_pert_qR) / eps_imag , self.rrL.T)

        return dSatLdqL, dSatLdqR, dSatRdqL, dSatRdqR

    def sat_der1_burgers_ec(self, q_fL, q_fR):
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

    def dfdq_sat_der1_burgers_ec(self, q_fL, q_fR):
        '''
        Purpose
        ----------
        Calculate the derivative of the entropy conservative SAT for Burgers equation.

        Parameters
        ----------
        q_fL : np array, shape (1,nelem)
            The extrapolated solution of the left element(s) to the facet(s).
        q_fR : np array, shape (1,nelem)
            The extrapolated solution of the left element(s) to the facet(s).

        Returns
        -------
        Given an interface, SatL and SatR are the contributions to either side
        qL are the solutions (NOT extrapolated) on either side. Therefore:
            dSatLdqL : derivative of the SAT in the left element wrt left solution
            dSatLdqR : derivative of the SAT in the left element wrt right solution
            dSatRdqL : derivative of the SAT in the right element wrt left solution
            dSatRdqR : derivative of the SAT in the right element wrt right solution
        '''

        dSatLdqL = fn.gs_lm((4*q_fL - q_fR)/6, self.rrR @ self.rrR.T)
        dSatLdqR = fn.gs_lm(-(q_fL + 2*q_fR)/6, self.rrR @ self.rrL.T)
        dSatRdqL = fn.gs_lm((q_fR + 2*q_fL)/6, self.rrL @ self.rrR.T)
        dSatRdqR = fn.gs_lm((q_fL - 4*q_fR)/6, self.rrL @ self.rrL.T)

        return dSatLdqL, dSatLdqR, dSatRdqL, dSatRdqR
    
