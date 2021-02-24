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


    ###################################### TODO ######################################
    
    def dfdq_sat_der1_upwind_scalar(self, q_fA, q_fB, sigma=1):
        '''
        Purpose
        ----------
        Calculate the derivative of the upwind SAT for a scalar equation.
        Used for implicit time marching.
        Parameters
        ----------
        q_fA : np array
            The extrapolated solution of the left element to the facet.
        q_fB : np array
            The extrapolated solution of the left element to the facet.
        sigma: float (default=1)
            if sigma=1, creates upwinding SAT
            if sigma=0, creates symmetric SAT
        Returns
        -------
        satA : np array
            The derivative of the contribution of the SAT for the first derivative to the element
            on the left.
        satB : np array
            The derivative of the contribution of the SAT for the first derivative to the element
            on the right.
        '''

        qf = (q_fA + q_fB)/2
        qf_diff = (q_fA - q_fB)[0]

        # First derivative of the flux wrt q
        A = self.diffeq.calc_dEdq(qf).todense()[0,0]

        A_eig = self.diffeq.calc_dEdq_eig(A) # Works for scalar equations
        sign_A_eig = np.sign(A_eig)

        factor_Ap = (1 + sign_A_eig*sigma)/2
        factor_An = (1 - sign_A_eig*sigma)/2

        A_upwind = factor_Ap * A
        A_downwind = factor_An * A

        d2Edq2 = self.diffeq.calc_d2Edq2(qf)[0,0]
        dAp_dq = factor_Ap * d2Edq2
        dAn_dq = factor_An * d2Edq2

        dSatA_dqA = self.rrR @ self.rrR.T * (dAn_dq * qf_diff/2 + A_downwind)
        dSatA_dqB = self.rrR @ self.rrL.T * (dAn_dq * qf_diff/2 - A_downwind)
        dSatB_dqA = self.rrL @ self.rrR.T * (dAp_dq * qf_diff/2 + A_upwind)
        dSatB_dqB = self.rrL @ self.rrL.T * (dAp_dq * qf_diff/2 - A_upwind)

        return dSatA_dqA, dSatA_dqB, dSatB_dqA, dSatB_dqB

    def dfdq_sat_der1_complexstep(self, qelemA, qelemB, eps_imag=1e-30):
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
        q_fA : np array
            The extrapolated solution of the left element to the facet.
        q_fB : np array
            The extrapolated solution of the left element to the facet.
        eps_imag : float, optional
            Size of the complex step
        Returns
        -------
        The derivative of the SAT contribution to the elements on both sides
        of the interface.
        '''

        q_fA, q_fB = self.get_interface_sol_unstruc(qelemA, qelemB)

        # Unperturbed case
        satA, satB = self.calc_sat(q_fA, q_fB)

        # Calculate perturbed matrices
        sat_len = satA.size
        neq_node = self.neq_node
        satA_pert_qA = np.zeros((sat_len, neq_node), dtype=complex)
        satA_pert_qB = np.zeros((sat_len, neq_node), dtype=complex)

        satB_pert_qA = np.zeros((sat_len, neq_node), dtype=complex)
        satB_pert_qB = np.zeros((sat_len, neq_node), dtype=complex)

        for i in range(neq_node):
            perturbation = np.zeros(neq_node, dtype=complex)
            perturbation[i] = eps_imag * 1j
            q_fA_pert = q_fA + perturbation
            q_fB_pert = q_fB + perturbation

            satA_pert_qA[:, i], satB_pert_qA[:, i] = self.calc_sat(q_fA_pert, q_fB)
            satA_pert_qB[:, i], satB_pert_qB[:, i] = self.calc_sat(q_fA, q_fB_pert)

        dSatA_dqA = np.imag(satA_pert_qA - satA[:, None]) /eps_imag
        dSatA_dqB = np.imag(satA_pert_qB - satA[:, None]) /eps_imag
        dSatB_dqA = np.imag(satB_pert_qA - satB[:, None]) /eps_imag
        dSatB_dqB = np.imag(satB_pert_qB - satB[:, None]) /eps_imag

        # Everything up to this step would work with differential equations
        # that have higher derivative terms. However, the chain rule that adds
        # the term rrR.T and rrL.T only applies to SATs for first derivatives
        # The complex step calculates dSAT_dq_fA and this adds the term
        # dq_fA_dqA = rrR.T. Together this gives dSAT_dqA, which is what we
        # want. The derivative wrt to qB is analogous.
        dSatA_dqA = dSatA_dqA @ self.rrR.T
        dSatA_dqB = dSatA_dqB @ self.rrL.T
        dSatB_dqA = dSatB_dqA @ self.rrR.T
        dSatB_dqB = dSatB_dqB @ self.rrL.T

        return dSatA_dqA, dSatA_dqB, dSatB_dqA, dSatB_dqB
