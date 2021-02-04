#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 09:19:58 2020

@author: andremarchildon
"""

import numpy as np
import Source.Methods.Functions as fn

'''
This file has two classes: SatBase and SatBaseCons. The latter inherits from
the former and is used for PDEs that are of the form dqdt + dEdx = G. For PDEs
that inherent SatBase but not SatBaseCons the methods calc_sat and dfdq_sat
must be specified. These indicate which of the method(s) are used to calculate
the SAT and its derivative. Look at the Ks1d.py file for an example on this.
'''

class SatBase:
    
    def set_sat(self, method):
        raise Exception('This base method should not be called')

    def calc_sat(self, *argv):
        raise Exception('This base method should not be called')

    def dfdq_sat(self, *argv):
        raise Exception('This base method should not be called')

    def dfds_sat(self, *argv):
        raise Exception('This base method should not be called')

    def get_interface_sol(self, qelem, is_left_of_facet):
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
            else:
                q_f = self.rrL.T @ qelem

        return q_f

    def sat_der1_upwind(self, q_fA, q_fB, sigma, avg='simple'):
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
        
        A = self.dEdq(qfacet)
        A_abs = self.dEdq_eig_abs(A)

        # Upwinding flux
        A_upwind = (A + sigma*A_abs)/2
        A_downwind = (A - sigma*A_abs)/2

        # Calculate the correction
        k = q_fA - q_fB

        # SAT for the left of the interface (SAT_N)
        satA = self.rrR @ fn.gm_gv(A_downwind, k)  # SAT for the left of the interface
        satB = self.rrL @ fn.gm_gv(A_upwind, k)    # SAT for the right of the interface

        return satA, satB
    
    def dfdq_sat_der1_upwind_scalar(self, q_fA, q_fB, sigma):
        '''
        Purpose
        ----------
        Calculate the derivative of the upwind SAT for a scalar equation.
        Used for implicit time marching.
        '''
        raise Exception('Not coded up yet.')


class SatBaseCons(SatBase):
    
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
            if self.neq_node == 1:
                self.sat_der1 = lambda qA,qB: self.sat_der1_upwind(qA, qB, 0)
                self.dfdq_sat_der1 = lambda qA,qB: self.dfdq_sat_der1_upwind_scalar(qA, qB, 0)
            else:
                self.sat_der1 = lambda qA,qB: self.sat_der1_upwind(qA, qB, 0) # use Roe average?
                self.dfdq_sat_der1 = lambda qA,qB: self.dfdq_sat_complexstep(qA, qB)
        elif method == 'upwind':
            if self.neq_node == 1:
                self.sat_der1 = lambda qA,qB: self.sat_der1_upwind(qA, qB, 1) 
                self.dfdq_sat_der1 = lambda qA,qB: self.self.dfdq_sat_der1_upwind_scalar(qA, qB, 1)
            else:
                self.sat_der1 = lambda qA,qB: self.sat_der1_upwind(qA, qB, 1) # use Roe average?
                self.dfdq_sat_der1 = lambda qA,qB: self.dfdq_sat_complexstep(qA, qB)
        else:
            raise Exception('Choice of SAT not understood.')

    def calc_sat(self, qelemA, qelemB):
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

        q_fA = self.get_interface_sol(qelemA, is_left_of_facet=True)
        q_fB = self.get_interface_sol(qelemB, is_left_of_facet=False)

        return self.sat_der1(q_fA, q_fB)

    def dfdq_sat(self, qelemA, qelemB):
        '''
        Purpose
        ----------
        Default function to calculate the derivative of the upwind SATs for a
        PDE with only first derivatives.

        Parameters
        ----------
        qelemA : np array
            The solution at the nodes in the element to the left of the interface.
        qelemB : np array
           The solution at the nodes in the element to the right of the interface.

        Returns
        -------
        The derivative of the SAT contribution to the elements on both sides
        of the interface.
        '''

        q_fA = self.get_interface_sol(qelemA, is_left_of_facet=True)
        q_fB = self.get_interface_sol(qelemB, is_left_of_facet=False)
        return self.dfdq_sat_der1(q_fA, q_fB)


    def dfdq_sat_complexstep(self, q_fA, q_fB, eps_imag=1e-30):
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
        
        print('WARNING: THIS IS NOT TESTED YET!')

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
    
    
""" Unused functions from SatBase:

    def dfdq_sat_der1_upwind_scalar(self, q_fA, q_fB, sigma=1):
        '''
        Purpose
        ----------
        Calculate the derivative of the upwind SAT for a scalar equation.
        Used for implicit time marching.

        Parameters
        ----------
        q_fA : np array
            The solution to the left of the facet.
        q_fB : np array
            The solution to the right of the facet.
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

        # First derivative of the flux wrt q
        A = self.dEdq(qf).todense()[0,0]
        A_abs = self.dEdq_eig_abs(A)
        A_upwind = (A + sigma*A_abs)/2
        A_downwind = (A - sigma*A_abs)/2

        # Second derivative of the flux wrt to q
        d2Edq2 = self.d2Edq2(qf)[0,0]
        if qf > 0:
            dAn_dq = 0      # \frac{dA^-}{dq}
            dAp_dq = d2Edq2 # \frac{dA^+}{dq}
        else:
            dAn_dq = d2Edq2 # \frac{dA^-}{dq}
            dAp_dq = 0      # \frac{dA^+}{dq}

        k = (q_fA - q_fB)[0]

        dSatA_dqA = self.rrR @ self.rrR.T * (dAn_dq * k/2 + A_downwind)
        dSatA_dqB = self.rrR @ self.rrL.T * (dAn_dq * k/2 - A_downwind)
        dSatB_dqA = self.rrL @ self.rrR.T * (dAp_dq * k/2 + A_upwind)
        dSatB_dqB = self.rrL @ self.rrL.T * (dAp_dq * k/2 - A_upwind)

        return dSatA_dqA, dSatA_dqB, dSatB_dqA, dSatB_dqB

    def sat_der2_scalar_linear(self, q_fA, q_fB, qder1_fA, qder1_fB):
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

        sigmaA = self.c_der2 /2
        sigmaB = -self.c_der2 /2

        satA = sigmaA * (self.der1.T @ self.rrR) * (q_fA - q_fB)
        satB = sigmaB * self.rrL * (qder1_fB - qder1_fA)

        return satA, satB

    def dfdq_sat_der2_scalar_linear(self, q_fA, q_fB, qder1_fA, qder1_fB):

        '''
        Derivative of the SAT for a linear scalar 2nd derivative term
        ie \frac{d^2 q}{dx^2}.
        See the function above (sat_der2_scalar_linear).
        '''

        sigmaA = self.c_der2 /2
        sigmaB = -self.c_der2 /2

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

    def sat_der4_scalar_linear(self, q_fA, q_fB, qder1_fA, qder1_fB, qder2_fA, qder2_fB, qder3_fA, qder3_fB):
        '''
        Purpose
        ----------
        Calculate the upwind SAT for a linear fourth derivative scalar term
        (ie \frac{d^4 q}{dx^4}). What the SAT weekly enforces on each side of
        the interface depends on whether the interface is at the boundary or
        not since the second and third derivatives are not available at
        boundaries.

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
        qder2_fA : np array
            The numerical second derivative of the solution at all the nodes
            in the element to the left of the interface.
        qder2_fB : np array
            The numerical second derivative of the solution at all the nodes
            in the element to the right of the interface.
        qder3_fA : np array
            The numerical third derivative of the solution at all the nodes
            in the element to the left of the interface.
        qder3_fB : np array
            The numerical third derivative of the solution at all the nodes
            in the element to the right of the interface.

        Returns
        -------
        satA : np array
            The contribution of the SAT for the second derivative to the
            element on the left.
        satB : np array
            The contribution of the SAT for the second  derivative to the
            element on the right.
        '''

        sigmaA1 = self.c_der4
        sigmaA2 = -self.c_der4
        sigmaB1 = -self.c_der4
        sigmaB2 = self.c_der4

        # True if the facet is the left booundary
        if (qder2_fA is None) or (qder3_fA is None):
            satA = None

            # Weekly enforce:   u and u_x to the right of the facet
            satB = sigmaB1 * (self.der3.T @ self.rrL) * (q_fB-q_fA) + sigmaB2 * (self.der2.T @ self.rrL) * (qder1_fB-qder1_fA)
        else:

            satA = sigmaA1 * (self.der3.T @ self.rrR) *(q_fA-q_fB) + sigmaA2 * (self.der2.T @ self.rrR) * (qder1_fA-qder1_fB)

            # True if the facet is the right booundary
            if (qder2_fB is None) or (qder3_fB is None):
                satB = None
            else:
                # Weekly enforce:   u_xx and u_xxx to the right of the facet
                satB = sigmaB1 * self.rrL * (qder3_fB-qder3_fA) + sigmaB2 * (self.der1.T @ self.rrL) * (qder2_fB-qder2_fA)

        return satA, satB

    def dfdq_sat_der4_scalar_linear(self, q_fA, q_fB, qder1_fA, qder1_fB, qder2_fA, qder2_fB, qder3_fA, qder3_fB):

        '''
        Derivative of the SAT for a linear scalar fourth derivative term
        ie \frac{d^4 q}{dx^4}.
        See the function above (sat_der4_scalar_linear).
        '''

        # True if the facet is the left booundary
        if (qder2_fA is None) or (qder3_fA is None):

            dSatA_dqA = None
            dSatA_dqB = None
            k = self.rrL @ self.rrR.T
            dSatB_dqA = self.c_der4 * (self.der3.T @ k - self.der2.T @ k @ self.der1)
            k = self.rrL @ self.rrL.T
            dSatB_dqB = self.c_der4 * (-self.der3.T @ k + self.der2.T @ k @ self.der1)
        else:
            k = self.rrR @ self.rrR.T
            dSatA_dqA = self.c_der4 * (self.der3.T @ k - self.der2.T @ k @ self.der1)
            k = self.rrR @ self.rrL.T
            dSatA_dqB = self.c_der4 * (-self.der3.T @ k + self.der2.T @ k @ self.der1)

            if (qder2_fB is None) or (qder3_fB is None):
                dSatB_dqA = None
                dSatB_dqB = None
            else:
                k = self.rrL @ self.rrR.T
                dSatB_dqA = self.c_der4 * (k @ self.der3 - self.der1.T @ k @ self.der2)
                k = self.rrL @ self.rrL.T
                dSatB_dqB = self.c_der4 * (-k @ self.der3 + self.der1.T @ k @ self.der2)

        return dSatA_dqA, dSatA_dqB, dSatB_dqA, dSatB_dqB


   
"""