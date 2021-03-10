#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 19:02:07 2021

@author: bercik
"""
import numpy as np
import Source.Methods.Functions as fn
#from Source.DiffEq.Quasi1DEulerA import build_F_vol, build_F_int

class SatDer1:


    def sat_der1_upwind(self, q_L, q_R, sigma=1, avg='simple'):
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
        q_fL, q_fR = self.get_interface_sol(q_L, q_R)
        
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
    
    def dfdq_sat_der1_upwind_scalar(self, q_L, q_R, sigma=1, avg='simple'):
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
        q_fL, q_fR = self.get_interface_sol(q_L, q_R)
        
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

    def dfdq_sat_der1_complexstep(self, q_L, q_R, eps_imag=1e-30):
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
        q_fL, q_fR = self.get_interface_sol(q_L, q_R)
        
        neq_node,nelem = q_fL.shape
        assert neq_node == self.neq_node,'neq_node does not match'
        satL_pert_qL = np.zeros((self.nen*neq_node,neq_node,nelem),dtype=complex)
        satR_pert_qL = np.zeros((self.nen*neq_node,neq_node,nelem),dtype=complex)
        satL_pert_qR = np.zeros((self.nen*neq_node,neq_node,nelem),dtype=complex)
        satR_pert_qR = np.zeros((self.nen*neq_node,neq_node,nelem),dtype=complex)
        
        for neq in range(neq_node):
            pert = np.zeros((self.neq_node,nelem),dtype=complex)
            pert[neq,:] = eps_imag * 1j
            satL_pert_qL[:,neq,:], satR_pert_qL[:,neq,:] = self.sat_der1(q_fL + pert, q_fR)
            satL_pert_qR[:,neq,:], satR_pert_qR[:,neq,:] = self.sat_der1(q_fL, q_fR + pert)
        
        dSatLdqL = fn.gm_lm(np.imag(satL_pert_qL) / eps_imag , self.rrR.T)
        dSatLdqR = fn.gm_lm(np.imag(satL_pert_qR) / eps_imag , self.rrL.T)
        dSatRdqL = fn.gm_lm(np.imag(satR_pert_qL) / eps_imag , self.rrR.T)
        dSatRdqR = fn.gm_lm(np.imag(satR_pert_qR) / eps_imag , self.rrL.T)

        return dSatLdqL, dSatLdqR, dSatRdqL, dSatRdqR

    def sat_der1_burgers_ec(self, q_L, q_R):
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
        q_fL, q_fR = self.get_interface_sol(q_L, q_R)
        
        qLqR = q_fL * q_fR
        qL2 = q_fL**2
        qR2 = q_fR**2

        satL = self.rrR @ (qL2/3 - qLqR/6 - qR2/6)    # SAT for the left of the interface
        satR = self.rrL @ (-qR2/3 + qLqR/6 + qL2/6)    # SAT for the right of the interface

        return satL, satR

    def dfdq_sat_der1_burgers_ec(self, q_L, q_R):
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
        q_fL, q_fR = self.get_interface_sol(q_L, q_R)

        dSatLdqL = fn.gs_lm((4*q_fL - q_fR)/6, self.rrR @ self.rrR.T)
        dSatLdqR = fn.gs_lm(-(q_fL + 2*q_fR)/6, self.rrR @ self.rrL.T)
        dSatRdqL = fn.gs_lm((q_fR + 2*q_fL)/6, self.rrL @ self.rrR.T)
        dSatRdqR = fn.gs_lm((q_fL - 4*q_fR)/6, self.rrL @ self.rrL.T)

        return dSatLdqL, dSatLdqR, dSatRdqL, dSatRdqR

    def sat_der1_crean_ec(self, q_L, q_R):
        '''
        Purpose
        ----------
        Calculate the SATs for the entropy consistent scheme by Crean et al 2018
        NOTE: ONLY WORKS FOR ELEMENTS WITH BOUNDARY NODES! Should use more
        general matrix formulations for other cases. See notes.
        
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
        q_fL, q_fR = self.get_interface_sol(q_L, q_R)
        
        #TODO: Rework Ismail_Roe to accept shapes (neq_node,nelem) rather than (neq_node,)
        neq,nelem = q_fL.shape
        numflux = np.zeros((neq,nelem))
        for e in range(nelem):
            numflux[:,e] = self.diffeq.ec_flux(q_fL[:,e], q_fR[:,e])
        
        satL = self.rrR @ ( self.diffeq.calcE(q_fL) - numflux )
        satR = self.rrL @ ( numflux - self.diffeq.calcE(q_fR) )
        
        #F_vol = build_F_vol(q, self.neq_node, self.diffeq.ec_flux)
        #build_F_int(q1, q2, neq, ec_flux)
        #build_F_vol(q, neq, ec_flux)

        return satL, satR   
    
    def sat_der1_crean_es(self, qL, qR):
        '''
        Purpose
        ----------
        Calculate the SATs for the entropy dissipative scheme by Crean et al 2018
        NOTE: ONLY WORKS FOR ELEMENTS WITH BOUNDARY NODES! Should use more
        general matrix formulations for other cases. See notes.
        
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
        q_fL, q_fR = self.get_interface_sol(qL, qR)
        
        #TODO: Rework Ismail_Roe to accept shapes (neq_node,nelem) rather than (neq_node,)
        neq,nelem = q_fL.shape
        numflux = np.zeros((neq,nelem))
        for e in range(nelem):
            numflux[:,e] = self.diffeq.ec_flux(q_fL[:,e], q_fR[:,e])
        
        qfacet = (q_fL + q_fR)/2 # Assumes simple averaging, can generalize
        # TODO: This will get all fucked up by svec
        rhoL, rhouL, eL = self.diffeq.decompose_q(qL)
        uL = rhouL / rhoL
        pL = (self.diffeq.g-1)*(eL - (rhoL * uL**2)/2)
        aL = np.sqrt(self.diffeq.g * pL/rhoL)
        rhoR, rhouR, eR = self.diffeq.decompose_q(qR)
        uR = rhouR / rhoR
        pR = (self.diffeq.g-1)*(eR - (rhoR * uR**2)/2)
        aR = np.sqrt(self.diffeq.g * pR/rhoR)
        LF_const = np.max([np.abs(uL)+aL,np.abs(uR)+aR],axis=(0,1))
        Lambda = self.diffeq.dqdw(qfacet)*LF_const
        
        w_fL= self.rrR.T @ self.diffeq.entropy_var(qL)
        w_fR= self.rrL.T @ self.diffeq.entropy_var(qR)
        
        satL = self.rrR @ ( self.diffeq.calcE(q_fL) - numflux - fn.gm_gv(Lambda, w_fL - w_fR))
        satR = self.rrL @ ( numflux - self.diffeq.calcE(q_fR) - fn.gm_gv(Lambda, w_fR - w_fL))

        return satL, satR 