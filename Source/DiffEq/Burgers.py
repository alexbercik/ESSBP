#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 12:41:26 2020

@author: bercik
"""

import numpy as np

from Source.DiffEq.DiffEqBase import PdeBase
import Source.Methods.Functions as fn
from numba import njit
from scipy.optimize import root_scalar, bisect

class Burgers(PdeBase):
    '''
    Purpose
    ----------
    This class provides the required functions to solve the Burgers equation
    '''

    # Diffeq info
    diffeq_name = 'Burgers'
    dim = 1
    neq_node = 1    # 1 equation in 1D
    eq_type = 'pde'
    pde_order = 1
    has_exa_sol = True

    def __init__(self, para=None, q0_type='SinWave',
                 use_split_form=False, split_alpha=2/3):

        super().__init__(para, q0_type)
        self.use_split_form = use_split_form
        self.split_alpha = split_alpha

    def exact_sol(self, time=0, x=None, guess=None):

        if x is None:
            x = self.x_elem

        reshape = False
        if x.ndim >1: 
            reshape=True
            orig_shape = x.shape
            x = x.flatten('F')

        u = np.empty_like(x) # initiate u

        if self.q0_type.lower() == 'sinwave':
            # manually set solution for initial condition u_0 = sin(2*pi*x)
            Tb = 1/(2*np.pi)
            u0 = lambda x0: np.sin(2*np.pi*x0)  # initial condition
            shock = 0.5  # by symmetry, the shock is at x=0.5 for t>= t_break
            eps = 1e-8 # A small step to define the second guess for the secant method.

            if time < Tb:
                # --- Pre-shock: single-valued solution ---
                for i, xi in enumerate(x):
                    # Solve f(x0) = x0 + sin(2*pi*x0)*t - xi = 0
                    f = lambda x0: x0 + u0(x0)*time - xi
                    # Choose initial guesses; one natural guess is
                    #guess0 = xi - u0(xi)*time
                    #guess1 = guess0 + eps  # a second, nearby guess
                    #sol = root_scalar(f, method='secant', x0=guess0, x1=guess1, xtol=1e-12, maxiter=1000)
                    if xi == 0: u[i] = 0
                    elif xi == 1: u[i] = 0
                    else:
                        sol = root_scalar(f, method='brentq', bracket=[eps, 1-eps], xtol=1e-12, maxiter=1000)
                        u[i] = u0(sol.root)
            else:
                # Post–shock: a shock forms at x=0.5.
                # for each spatial point x, invert the characteristic relation using the correct branch.
                for i, xi in enumerate(x):
                    if xi == 0: u[i] = 0
                    elif xi == 1: u[i] = 0
                    else:
                        if abs(xi - shock) < 1e-12:
                            # At the shock, assign the Rankine–Hugoniot value.
                            u[i] = 0.0
                        elif abs(xi - 0.0) < 1e-12 or abs(xi - 1.0) < 1e-12:
                            # At the endpoints, just set u to zero
                            u[i] = 0.0
                        elif xi < shock:
                            # For x to the left of the shock, invert f(x0)= x0 + sin(2*pi*x0)*t - xi on x0 in [0, shock].
                            f = lambda x0: x0 + u0(x0)*time - xi
                            # f(0)= -xi (negative) and f(shock)= shock + sin(pi)*t - xi = 0.5 - xi (positive for 0<xi<0.5)
                            #if f(eps)*f(shock-eps) > 0:
                            #    # If the function has the same sign at both endpoints, we need to use a different bracket.
                            #    print('WARNING: The function has the same sign at both endpoints.')
                            #    print(f(eps), f(shock-eps), xi, time)
                            sol = root_scalar(f, method='brentq', bracket=[eps, shock-eps], xtol=1e-12, maxiter=1000)
                            u[i] = u0(sol.root)
                        else:
                            # For x to the right of the shock, invert on x0 in [shock, 1].
                            f = lambda x0: x0 + u0(x0)*time - xi
                            # f(shock)= 0.5 - xi (negative for xi>0.5) and f(1)= 1 - xi (positive for 0.5<xi<1)
                            #if f(eps)*f(shock-eps) > 0:
                            #    # If the function has the same sign at both endpoints, we need to use a different bracket.
                            #    print('WARNING: The function has the same sign at both endpoints.')
                            #    print(f(eps), f(shock-eps), xi, time)
                            sol = root_scalar(f, method='brentq', bracket=[shock+eps, 1-eps], xtol=1e-12, maxiter=1000)
                            u[i] = u0(sol.root)

        else:

            Tb = self.calc_breaking_time(print_res=False)
            if time > Tb:
                print(f'WARNING: Time {time:.3g} is greater than the breaking time {Tb:.3g}.')
                print('          Ignoring analytical solution because jump conditions are not yet implemented.')
                u = np.zeros_like(x)
            
            else:

                #u0 = self.set_q0(xy=x)
                #a,b = np.min(u0), np.max(u0) # u(x,t) must be some u0(x), so bounded by min & max
                #a,b = a-(b-a)/20 , b+(b-a)/20 # expand the boundaries slightly just in case
                for i in range(len(x)):
                    # the below is more efficient, but it sometimes fails with periodic boundaries
                    # because the endpoints do not have opposite signs (they are equal)
                    # Define the periodic version of the initial condition
                    #f = lambda z: self.set_q0(xy=(np.mod(z - self.xmin, self.dom_len) + self.xmin))
                    # Define the equation whose root we are seeking.
                    # Here, we also ensure that the argument (x[i] - z*time) is wrapped into the periodic domain.
                    #eq = lambda z: f(np.mod(x[i] - z*time - self.xmin, self.dom_len) + self.xmin) - z
                    # Now use a root-finding method (like bisection) on eq.
                    #u[i] = bisect(eq, self.xmin - 0.01, self.xmax + 0.01, xtol=1e-12, maxiter=1000)
                    
                    u0 = lambda x0 : self.set_q0(xy=x0)
                    modx = lambda x0: np.mod(x0-self.xmin,self.dom_len) + self.xmin
                    eq = lambda x0 : modx(x[i] - u0(x0)*time) - x0
                    #eq = lambda x0 : x[i] - u0(x0)*time - x0
                    if guess is None:
                        u_0 = u0(x[i]-u0(x[i])*time)
                    else:
                        u_0 = guess[i] # u0(x0) = u(x,t)
                    res = root_scalar(eq,bracket=[self.xmin-0.01,self.xmax+0.01],method='secant',
                                    x0=modx(x[i]-u_0*time),xtol=1e-12,maxiter=1000)
                    x0 = res.root
                    u[i] = u0(x0)
            
        if reshape:
            u = np.reshape(u,orig_shape,'F')

        return u

    def calcEx(self, q):
        E = 0.5*q**2
        return E
    
    def nonconservative_coeff(self, q):
        return q

    def dExdx(self, q, E):

        if self.use_split_form:
            dExdx = (self.split_alpha/2.) * self.gm_gv(self.Dx, q**2) \
                  + (1.-self.split_alpha) * q * self.gm_gv(self.Dx, q)
        else:
            dExdx = self.gm_gv(self.Dx, E)

        return dExdx

    def dExdq(self, q):

        dExdq = fn.gdiag_to_gbdiag(q)
        return dExdq
    
    def d2Exdq2(self, q):
        
        d2Exdq2 = fn.gdiag_to_gm(np.ones(q.shape))
        return d2Exdq2

    def dfdq(self, q):
        # take dExdx as a vector a_i(q) and find matrix d(a_i)/d(q_j)

        if self.use_split_form:
            # these both do the same, but the second is a bit faster
            #dfdq = -(1/3)*(2*fn.lm_gm(self.der1,fn.gdiag_to_gm(q)) + fn.gdiag_to_gm(self.der1@q) + fn.gm_lm(fn.gdiag_to_gm(q),self.der1))
            #dfdq = -(1/3)*(2*np.multiply(self.der1[:,:,None],q) + fn.gdiag_to_gm(self.der1 @ q) + fn.gm_lm(fn.gdiag_to_gm(q),self.der1))
            dfdq = -self.split_alpha*fn.gm_gv_colmultiply(self.Dx,q) - (1-self.split_alpha)*(fn.gdiag_to_gm(fn.gm_gv(self.Dx, q)) + fn.gdiag_gm(q,self.Dx))
        else:
            # this does the same as the base function, just a bit faster
            dfdq = - fn.gm_gv_colmultiply(self.Dx,q)
            
        return dfdq

    def calc_LF_const(self):
        ''' Constant for the Lax-Friedrichs flux. Not needed for SBP.'''
        q = fn.check_q_shape(self.set_q0())
        return np.max(np.abs(q))

    def dExdq_abs(self, q, entropy_fix):

        dExdq_abs = fn.gdiag_to_gbdiag(abs(q))
        return dExdq_abs
    
    def maxeig_dExdq(self, q):
        ''' return the absolute maximum eigenvalue - used for LF fluxes '''
        return np.abs(q)
    
    def maxeig_dEndq(self, q, dxidx):
        ''' return the absolute maximum eigenvalue - used for LF fluxes '''
        return np.abs(q*dxidx)
    
    def entropy_var(self,q):
        return q
    
    def dqdw(self,q):
        return fn.gdiag_to_gm(np.ones(q.shape))
    
    def maxeig_dqdw(self,q):
        return np.ones(q.shape)
    
    def dExdw_abs(self, q, entropy_fix):

        dExdw_abs = fn.gdiag_to_gbdiag(abs(q))
        return dExdw_abs
    
    def entropy(self,q):
        e = q**2/2
        return e
    
    def a_energy(self,q):
        ''' compute the global U-norm SBP energy of global solution vector q '''
        return np.tensordot(q, q * self.H * q)
    
    def a_energy_der(self,q,dqdt):
        ''' compute the global U-norm SBP energy derivatve of global solution vector q '''
        return 2 * np.tensordot(q, q * self.H * dqdt)
    
    def a_conservation(self,q):
        ''' compute the global A-conservation SBP of global solution vector q, equal to entropy/energy '''
        return np.sum(q * self.H * q)
    
    def calc_breaking_time(self,sig_fig=3,print_res=True):
        ''' estimate the time at which the solution breaks '''
        q0 = self.set_q0()
        dqdx = self.gm_gv(self.Dx, q0)
        Tb = -1/np.min(dqdx)
        if print_res:
            print(f'The breaking time is approximately T = {Tb:.{sig_fig}g}')
        else:
            return Tb
    
    @njit
    def ec_flux(qL,qR):
        ''' entropy conservative 2-point flux for the Hadamard form '''
        fx = (qL**2 + qL*qR + qR**2)/6
        return fx
    
    @njit   
    def central_flux(qL,qR):
        ''' a central 2-point flux for hadamard form.
        This allows us to jit the hadamard flux functions. '''
        f = fn.arith_mean(qL**2/2,qR**2/2)
        return f

