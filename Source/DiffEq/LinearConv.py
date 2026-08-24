#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 13:16:51 2020

@author: bercik
"""

import numpy as np

from Source.DiffEq.DiffEqBase import PdeBase
import Source.Methods.Functions as fn
from numba import njit

class LinearConv(PdeBase):
    '''
    Purpose
    ----------
    This class provides the required functions to solve the linear convection
    equation:
    '''

    diffeq_name = 'LinearConvection'
    dim = 1
    neq_node = 1    # 1 equation in 1D
    pde_order1 = True
    pde_order2 = False
    x = None
    has_exa_sol = True
    para_names = ('a',)
    a_fix = 1
    para_fix = [a_fix]

    def __init__(self, para, q0_type='SinWave', had_flux='central',
                 arcsinh_ureg=0.1, arcsinh_eps=0.0,
                 logreg_ureg=0.1, logreg_m=2):

        super().__init__(para, q0_type)
        self.a = self.para[0]
        self._arcsinh_ureg = float(arcsinh_ureg)
        self._arcsinh_eps = float(arcsinh_eps)
        self._logreg_ureg = float(logreg_ureg)
        self._logreg_m = int(logreg_m)
        
        if self.a == self.a_fix:
            print('Using the fixed a={} diffeq functions since params match.'.format(self.a_fix))
            def dExdq_fix(q):
                nen,nelem = np.shape(q)
                return np.ones((nen,1,1,nelem),dtype=q.dtype)
            self.dExdq = dExdq_fix
            self.dExdq_abs = dExdq_fix
            self.maxeig_dExdq = lambda q : np.ones(q.shape)
            self.central_flux = self.central_fix_flux
            self.logarithmic_flux = self.logarithmic_fix_flux
            self.geometric_flux = self.geometric_fix_flux
            self.harmonic_flux = self.harmonic_fix_flux
            self.logreg_flux = self.logreg_fix_flux
        
        if had_flux.lower() == 'central':
            self.entropy = self.entropy_central
            self.entropy_var = self.entropy_var_central
            self.dqdw = self.dqdw_central
            self.maxeig_dqdw = self.maxeig_dqdw_central
        elif had_flux.lower() == 'logarithmic':
            self.entropy = self.entropy_log
            self.entropy_var = self.entropy_var_log
            self.dqdw = self.dqdw_log
            self.maxeig_dqdw = self.maxeig_dqdw_log
        elif had_flux.lower() == 'geometric':
            self.entropy = self.entropy_geom
            self.entropy_var = self.entropy_var_geom
            self.dqdw = self.dqdw_geom
            self.maxeig_dqdw = self.maxeig_dqdw_geom
        elif had_flux.lower() == 'arcsinh':
            self.entropy = self.entropy_arcsinh
            self.entropy_var = self.entropy_var_arcsinh
            self.dqdw = self.dqdw_arcsinh
            self.maxeig_dqdw = self.maxeig_dqdw_arcsinh
            self._bind_arcsinh_flux()
        elif had_flux.lower() == 'harmonic':
            self.entropy = self.entropy_harm
            self.entropy_var = self.entropy_var_harm
            self.dqdw = self.dqdw_harm
            self.maxeig_dqdw = self.maxeig_dqdw_harm
        elif had_flux.lower() == 'logreg':
            self.entropy = self.entropy_logreg
            self.entropy_var = self.entropy_var_logreg
            self.dqdw = self.dqdw_logreg
            self.maxeig_dqdw = self.maxeig_dqdw_logreg
            self._bind_logreg_flux()
        else:
            raise ValueError(f'Invalid had_flux: {had_flux}. Must be "central", "logarithmic", "geometric", "harmonic", "arcsinh", or "logreg".')


    def exact_sol(self, time=0, x=None, **kwargs):

        if x is None:
            x = self.x_elem

        x_mod = np.mod((x - self.xmin) - self.a*time, self.dom_len) + self.xmin
        exa_sol = self.set_q0(xy=x_mod)

        return exa_sol

    def dfdq(self, q):
        # I of course don't need this but it speeds things up a bit 
        # compared to the default function since it's so simple

        dfdq = - self.a * self.Dx
        return dfdq

    def calcEx(self, q):

        E = self.a * q
        return E
    
    def nonconservative_coeff(self, q):
        return self.a

    def dExdq(self, q):
        nen,nelem = np.shape(q)
        dExdq = self.a*np.ones((nen,1,1,nelem),dtype=q.dtype)
        return dExdq
    
    def dEndq(self, q, dxidx):
        nen,nelem = np.shape(q)
        dExdq = self.a*dxidx*np.ones((nen,1,1,nelem),dtype=q.dtype)
        return dExdq

    def d2Exdq2(self, q):
        nen,nelem = np.shape(q)
        dExdq = self.a*np.zeros((nen,1,1,nelem),dtype=q.dtype)
        return dExdq

    def dExdq_abs(self, q, entropy_fix=False):
        nen,nelem = np.shape(q)
        dExdq = abs(self.a)*np.ones((nen,1,1,nelem),dtype=q.dtype)
        return dExdq

    def maxeig_dExdq(self, q):
        ''' return the absolute maximum eigenvalue - used for LF fluxes '''
        maxeig = np.ones(q.shape)*abs(self.a)
        # this is actually slower: np.ones_like(qf_avg)*self.a
        return maxeig
    
    def maxeig_dEndq(self, q, dxidx):
        ''' return the absolute maximum eigenvalue - used for LF fluxes '''
        maxeig = abs(dxidx*self.a)
        return maxeig
    

    def calc_LF_const(self,xy):
        ''' Constant for the Lax-Friedrichs flux'''
        return abs(self.a)

    @njit   
    def central_fix_flux(qL,qR):
        ''' a central 2-point flux for hadamard form but with a fixed at 1.
        This allows us to jit the hadamard flux functions. '''
        f = fn.arith_mean(qL,qR)
        return f

    @njit   
    def logarithmic_fix_flux(qL,qR):
        ''' a logarithmic 2-point flux for hadamard form but with a fixed at 1.
        This allows us to jit the hadamard flux functions. '''
        f = fn.log_mean(qL,qR)
        return f

    @njit   
    def geometric_fix_flux(qL,qR):
        ''' a geometric 2-point flux for hadamard form but with a fixed at 1.
        This allows us to jit the hadamard flux functions. '''
        f = fn.geom_mean(qL,qR)
        return f

    @njit
    def arcsinh_fix_flux(qL, qR, ureg, eps=0.0):
        ''' a regularized version of the logarithmic 2-point flux for hadamard form but with a fixed at 1.
        This allows us to jit the hadamard flux functions. '''
        return fn.arcsinh_mean(qL, qR, ureg, eps)

    @njit
    def harmonic_fix_flux(qL, qR):
        '''A harmonic 2-point flux for hadamard form but with a fixed at 1.'''
        f = fn.harm_mean(qL, qR)
        return f

    @njit
    def logreg_fix_flux(qL, qR, ureg, m):
        """
        Regularized logarithmic 2-point flux for Hadamard form
        with the advection speed fixed at 1.
        """
        return fn.logreg_mean(qL, qR, ureg, m)

    def entropy_central(self, q):
        ''' nodal values of the entropy for the central flux '''
        return q*q
    
    def entropy_var_central(self, q):
        ''' nodal values of the entropy variables w(q) for the central flux '''
        return q

    def dqdw_central(self,q):
        ''' hessian P of potential phi wrt entropy variables w '''
        return fn.gdiag_to_gbdiag(np.ones_like(q))
    
    def maxeig_dqdw_central(self,q):
        ''' maximum eigenvalues of hessian P, i.e. abs(dqdw) for scalar '''
        return np.ones(q.shape)

    def entropy_geom(self, q):
        ''' nodal values of the entropy for the central flux '''
        return -2*np.sqrt(q)
    
    def entropy_var_geom(self, q):
        ''' nodal values of the entropy variables w(q) for the central flux '''
        return -1/np.sqrt(q)

    def dqdw_geom(self,q):
        ''' hessian P of potential phi wrt entropy variables w '''
        return fn.gdiag_to_gbdiag(2*q*np.sqrt(q))
    
    def maxeig_dqdw_geom(self,q):
        ''' maximum eigenvalues of hessian P, i.e. abs(dqdw) for scalar '''
        return 2*np.abs(q*np.sqrt(q))

    def entropy_log(self, q):
        ''' nodal values of the entropy for the logarithmic flux '''
        return q*np.log(q) - q
    
    def entropy_var_log(self, q):
        ''' nodal values of the entropy variables w(q) for the logarithmic flux '''
        return np.log(q)

    def dqdw_log(self,q):
        ''' hessian P of potential phi wrt entropy variables w '''
        return fn.gdiag_to_gbdiag(q)
    
    def maxeig_dqdw_log(self,q):
        ''' maximum eigenvalues of hessian P, i.e. abs(dqdw) for scalar '''
        return np.abs(q)

    def entropy_arcsinh(self, q):
        ''' nodal values of the entropy for the arcsinh flux '''
        ureg = self.arcsinh_ureg
        qe = q - self._arcsinh_eps
        return qe*np.arcsinh(qe/ureg) - np.sqrt(qe*qe + ureg*ureg) + ureg

    def entropy_var_arcsinh(self, q):
        ''' nodal values of the entropy variables w(q) for the arcsinh flux '''
        return np.arcsinh((q - self._arcsinh_eps)/self.arcsinh_ureg)

    def dqdw_arcsinh(self, q):
        ''' hessian P = dq/dw for the arcsinh flux '''
        qe = q - self._arcsinh_eps
        ureg = self.arcsinh_ureg
        p = np.sqrt(qe*qe + ureg*ureg)
        return fn.gdiag_to_gbdiag(p)

    def maxeig_dqdw_arcsinh(self, q):
        ''' maximum eigenvalues of hessian P, i.e. abs(dq/dw) for scalar '''
        qe = q - self._arcsinh_eps
        ureg = self.arcsinh_ureg
        return np.sqrt(qe*qe + ureg*ureg)

    def entropy_harm(self, q):
        '''Nodal values of the entropy for the harmonic flux.'''
        return 0.5 / q

    def entropy_var_harm(self, q):
        '''Nodal values of the entropy variables w(q) for the harmonic flux.'''
        return -0.5 / (q*q)

    def dqdw_harm(self, q):
        '''Hessian P = dq/dw for the harmonic flux.'''
        p = q*q*q
        return fn.gdiag_to_gbdiag(p)

    def maxeig_dqdw_harm(self, q):
        '''Maximum eigenvalues of Hessian P, i.e. abs(dq/dw) for scalar.'''
        return np.abs(q*q*q)

    def entropy_var_logreg(self, q):
        """Nodal values of the entropy variables w(q) for the logreg flux."""
        ureg = self.logreg_ureg
        m = self.logreg_m
        r = ureg / (q + ureg)
        w = np.log1p(q / ureg)
        rpow = r
        for j in range(1, m):
            w += (1.0 / j) * (1.0 - rpow)
            rpow = rpow * r
        return w


    def entropy_logreg(self, q):
        """Nodal values of the entropy for the logreg flux."""

        ureg = self.logreg_ureg
        m = self.logreg_m
        r = ureg / (q + ureg)
        w = np.log1p(q / ureg)
        rpow = r
        for j in range(1, m):
            w += (1.0 / j) * (1.0 - rpow)
            rpow = rpow * r
        psi = (
            q
            - ureg / (m - 1)
            * (1.0 - r**(m - 1))
        )
        # U = q*w - psi
        return q * w - psi


    def dqdw_logreg(self, q):
        """Hessian P = dq/dw for the logreg flux."""
        ureg = self.logreg_ureg
        m = self.logreg_m
        r = ureg / (q + ureg)
        # dw/dq = 1/(q+ureg) * sum_{j=0}^{m-1} r**j
        s = np.ones_like(q)
        rpow = r
        for j in range(1, m):
            s += rpow
            rpow = rpow * r
        p = (q + ureg) / s
        return fn.gdiag_to_gbdiag(p)


    def maxeig_dqdw_logreg(self, q):
        """Maximum eigenvalues of Hessian P, i.e. abs(dq/dw) for scalar."""
        ureg = self.logreg_ureg
        m = self.logreg_m
        r = ureg / (q + ureg)
        s = np.ones_like(q)
        rpow = r
        for j in range(1, m):
            s += rpow
            rpow = rpow * r
        p = (q + ureg) / s
        return np.abs(p)


    @property
    def arcsinh_ureg(self):
        return self._arcsinh_ureg

    @arcsinh_ureg.setter
    def arcsinh_ureg(self, value):
        """Set the arcsinh smoothing width and rebind the jitted 2-point flux.

        Must be set before the SBP solver is constructed (the solver snapshots
        diffeq.arcsinh_flux as calc_had_flux).
        """
        self._arcsinh_ureg = float(value)
        self._bind_arcsinh_flux()

    def _bind_arcsinh_flux(self):
        """Capture ureg/eps in a 2-arg njit closure so Hadamard kernels stay compiled."""
        ureg = float(self._arcsinh_ureg)
        eps = float(self._arcsinh_eps)

        @njit
        def arcsinh_flux(qL, qR):
            return fn.arcsinh_mean(qL, qR, ureg, eps)

        self.arcsinh_flux = arcsinh_flux

    @property
    def logreg_ureg(self):
        return self._logreg_ureg

    @logreg_ureg.setter
    def logreg_ureg(self, value):
        """Set the logreg regularization scale and rebind the jitted 2-point flux.

        Must be set before the SBP solver is constructed (the solver snapshots
        diffeq.logreg_flux as calc_had_flux).
        """
        self._logreg_ureg = float(value)
        self._bind_logreg_flux()

    @property
    def logreg_m(self):
        return self._logreg_m

    @logreg_m.setter
    def logreg_m(self, value):
        """Set the logreg regularization order and rebind the jitted 2-point flux.

        Must be an integer >= 2 and must be set before the SBP solver is
        constructed.
        """
        value = int(value)

        if value < 2:
            raise ValueError("logreg_m must be an integer >= 2.")

        self._logreg_m = value
        self._bind_logreg_flux()


    def _bind_logreg_flux(self):
        """Capture ureg/m in a 2-arg njit closure so Hadamard kernels stay compiled."""

        # Allow either parameter to be initialized first.
        if not hasattr(self, "_logreg_ureg") or not hasattr(self, "_logreg_m"):
            return

        ureg = float(self._logreg_ureg)
        m = int(self._logreg_m)

        @njit
        def logreg_flux(qL, qR):
            return fn.logreg_mean(qL, qR, ureg, m)

        self.logreg_flux = logreg_flux