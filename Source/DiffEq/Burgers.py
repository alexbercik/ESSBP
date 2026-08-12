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
from scipy.optimize import root_scalar, newton, minimize_scalar
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


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
    pde_order1 = True
    pde_order2 = False
    has_exa_sol = True
    u0min = None # minimim of IC
    u0max = None # maximum of IC
    tb = None # breaking time
    xb = None # breaking location
    x0b = None # initial x0 of the breaking characteristic
    tfix_initial_time = None

    def __init__(self, para=None, q0_type='SinWave',
                 use_split_form=False, split_alpha=2/3):

        super().__init__(para, q0_type)
        self.use_split_form = use_split_form
        self.split_alpha = split_alpha

    def eq_x(self, x0, t):
        " given initial x0 and t, get the x based on the characteristics"
        return x0 + self.u0(x0)*t
    
    def eq_x0(self, x0, x, t):
        " implicit equation relating x0, t, and x based on the characteristics"
        return x0 + self.u0(x0)*t - x
    
    def eq_tb(self, xb0, tb):
        " implicit equation relating xb0 and tb based on the characteristics"
        " where xb0 is the initial x0 of a characteristic, and tb is the breaking time of that charaacteristic"
        if getattr(self, '_exact_sol_preshock_only', False):
            raise NotImplementedError(
                'Burgers post-shock characteristic construction needs analytic du0/dx; '
                'this initial condition only supports exact_sol for t <= tb (see PdeBase IC path).'
            )
        return tb + 1/self.du0dx(xb0)
    
    def modx(self, x):
        " useful for periodic boundary condition, loops x around the domain"
        return np.mod(x-self.xmin,self.dom_len) + self.xmin

    def _breaking_slip(self, t, *, tb_for_slip=None):
        '''Float tolerance band near breaking time (matches former slip_* formulas).'''
        t = abs(float(np.asarray(t, dtype=float).reshape(-1)[0]))
        if tb_for_slip is None:
            if self.tb is not None and np.isfinite(float(self.tb)):
                tb_abs = abs(float(self.tb))
            else:
                tb_abs = 0.0
        else:
            tb_abs = abs(float(tb_for_slip))
        return max(
            1e-9,
            np.finfo(float).eps * (t + tb_abs + 1.0) * 64.0,
            1e-9 * float(self.dom_len),
        )

    @staticmethod
    def _char_tb_is_zero(tb):
        return tb is not None and np.isfinite(float(tb)) and abs(float(tb)) < 1e-14

    def _assert_char_exact_for_tb_zero(self, t, *, context):
        t0 = abs(float(np.asarray(t, dtype=float).reshape(-1)[0]))
        slip = self._breaking_slip(t0, tb_for_slip=0.0)
        assert t0 <= slip, (
            f'Burgers {context}: the classical characteristic exact solution is not valid for this '
            'problem because tb == 0 (e.g. piecewise-constant coarse Gassner-type initial data). '
            'Use |t| within O(slip) of 0 (same tolerance as t <= tb near breaking), or do not use '
            'exact_sol / find_x0 / find_envelope for larger t.'
        )
    
    def find_x0(self,x,t,N=None,print_warnings=True,limit_sols=True):
        " given a point (x,t) in characterstic space, find the x0 that feed into it"
        dom = (x - self.u0max*t - 0.01, x - self.u0min*t + 0.01)
        if N is None:
            if getattr(self, '_exact_sol_preshock_only', False) and self._char_tb_is_zero(self.tb):
                self._assert_char_exact_for_tb_zero(t, context='find_x0')
                N = 30
            else:
                N = max(30,int(t/self.tb*30))
        elif N > 1e6: raise Exception(f'N is too large. N={N}, t={t}, x={x}')
        x0s = self.find_all_roots(self.eq_x0, (x,t), domain=dom, N=N)
        if limit_sols:
            # keep only the x0 points that lead to x being in the domain at time t
            # should not do this for the shock location, since it moves outside the domain
            xs = self.eq_x(x0s, t)
            x0s = x0s[(xs >= self.xmin - 1e-8) & (xs <= self.xmax + 1e-8)]
        if len(x0s) == 3:
            # ignore the middle one
            # note that if the flow is negative, the left and right roots will be swapped, but this shouldn't matter
            return [x0s[0], x0s[2]]
        elif len(x0s) == 5:
            # ignore the middle and outer ones
            return [x0s[1], x0s[3]]
        elif len(x0s) == 7:
            # ignore the middle and outer ones
            return [x0s[2], x0s[4]]
        elif len(x0s) == 1:
            return x0s
        elif len(x0s) in (2, 4, 6):
            # Even count: recurse with larger N forever if the search keeps the same multiplicity.
            slip = self._breaking_slip(t)
            preshock_only = getattr(self, '_exact_sol_preshock_only', False)
            tb_ok = self.tb is not None and np.isfinite(float(self.tb))
            in_preshock_window = (not tb_ok) or (t <= float(self.tb) + slip)
            if preshock_only and in_preshock_window:
                xa = np.asarray(x0s, dtype=float)
                xs = np.asarray(self.eq_x(xa, t), dtype=float)
                xq = float(np.asarray(x).reshape(-1)[0])
                d = np.abs(self.modx(xs) - self.modx(xq))
                d = np.minimum(d, float(self.dom_len) - d)
                return np.array([xa[int(np.argmin(d))]])
            if len(x0s) == 2:
                return [float(x0s[0]), float(x0s[1])]
            if print_warnings:
                print(f'{len(x0s)} roots found for (x,t)={(x,t)}. Increasing N to {11*N}')
            return self.find_x0(x, t, N=11 * N, print_warnings=print_warnings, limit_sols=limit_sols)
        else:
            if print_warnings:
                if len(x0s) > 0:
                    print(f'{len(x0s)} roots found for (x,t)={(x,t)}. Increasing N to {11*N}')
                else:
                    print(f'No root found for (x,t)={(x,t)}. Increasing N to {11*N}')
            return self.find_x0(x, t, N=11 * N, print_warnings=print_warnings, limit_sols=limit_sols)
        
    def find_envelope(self,t,N=None,print_warnings=True):
        " given a time t, find the two x0 that define the `shocked envelope' "
        if getattr(self, '_exact_sol_preshock_only', False) and np.isfinite(self.tb):
            slip = self._breaking_slip(t)
            if t > self.tb + slip:
                raise NotImplementedError(
                    'Burgers find_envelope for t > tb requires analytic du0/dx; '
                    'this initial condition only supports exact_sol for t <= tb.'
                )
        if t <= self.tb: return None, None
        if N is None:
            if getattr(self, '_exact_sol_preshock_only', False) and self._char_tb_is_zero(self.tb):
                self._assert_char_exact_for_tb_zero(t, context='find_envelope')
                N = 30
            else:
                N = max(30,int(t/self.tb*30))
        elif N > 1e6: raise Exception('N is too large.', N)
        dom = (self.xmin - self.u0max*t - 0.01, self.xmax - self.u0min*t + 0.01)
        x0s = self.find_all_roots(self.eq_tb, t, domain=dom, N=N)
        if len(x0s) > 0:
            x0s = np.array(x0s)
            # keep only the x0 points that lead to x being in the domain at time t
            xs = self.eq_x(x0s, t)
            x0s = x0s[(xs >= self.xmin) & (xs <= self.xmax)]
            if len(x0s) == 2:
                return x0s
            else:
                # theoretically there could be 2 different shocks... should maybe generalize this?
                raise Exception('Incorrect number of roots found',t,x0s)
        else:
            if print_warnings:
                print(f'No root found for t={t}. Increasing N to {10*N}')
            res = self.find_envelope(t,N=10*N,print_warnings=print_warnings)
            return res
        
    def dsxdt(self,t,sx):
        ''' calculates the shock speed via Rankine-Hugoniot'''
        if t == self.tb: return [self.u0(self.x0b)]
        elif t < self.tb: return [0]
        else:
            # first get the x0 that correspond to the right and left characteristics at sx
            x0 = []
            N = 3
            while len(x0) < 2:
                N *= 11
                x0 = self.find_x0(sx[0],t,N=N,limit_sols=False)
                if N > 1e6:
                    lims = np.sort(self.eq_x(self.find_envelope(t),t))
                    raise Exception(f'N={N} is too large. t={t}, sx={sx[0]}, envelope={lims}')
            uL, uR = self.u0(x0[0]), self.u0(x0[1])
            res = 0.5*(uL + uR)
            return [res]

    def exact_sol(self, time=0, x=None, guess=None, **kwargs):
        # NOTE: Assumes the PDE has periodic BCs

        if x is None:
            x = self.x_elem

        if self.tb is None:
            # calc_breaking_time returns discrete Tb but must not overwrite coarse IC self.tb == 0.
            self.calc_breaking_time(print_res=False)

        slip_exact = self._breaking_slip(time)

        if getattr(self, '_exact_sol_preshock_only', False) and np.isfinite(self.tb):
            if time > float(self.tb) + slip_exact:
                raise NotImplementedError(
                    'Burgers.exact_sol for this initial condition is only implemented for t <= tb '
                    '(PdeBase IC path without analytic du0/dx).'
                )
            if self._char_tb_is_zero(self.tb):
                self._assert_char_exact_for_tb_zero(time, context='exact_sol')
        
        if guess is not None:
            assert(guess.shape == x.shape),'guess must be the same shape as x'

        reshape = False
        if x.ndim >1: 
            reshape=True
            orig_shape = x.shape
            x = x.flatten('F')

        u = np.empty_like(x) # initiate u

        if time > float(self.tb) + slip_exact:
            # now solve the ODE for the shock location xs
            tm = solve_ivp(self.dsxdt, (self.tb, time), [self.xb], method='RK45', rtol=3e-13, atol=3e-13, max_step=1e-1)
            sx = self.modx(tm.y[0,-1])
            x0L, x0R = self.find_x0(sx,time,limit_sols=False)
            x0LR, x0RL = x0R - (self.xmax-self.xmin), x0L + (self.xmax-self.xmin) # periodic BC, shocks on left and right


            for i in range(len(x)):
                x0 = self.find_x0(x[i],time)

                if len(x0)==1:
                    assert (x0[0] < x0L) or (x0[0] > x0R), 'x0 should be outside of the shocked envelope'
                    u[i] = self.u0(x0[0])
                elif x0[0] < x0LR: # in shock envelope of the shock on the left, on the right
                    u[i] = self.u0(x0[1])
                elif x0[0] < x0L: # in shock envelope of main shock, but from the left
                    u[i] = self.u0(x0[0])
                elif abs(self.eq_x(x0[0],time) - sx) < 1e-12: 
                    u[i] = self.u0(self.x0b)
                elif (x0[1] > x0R) and (x0[1] < x0RL): # in shock envelop of the main shock, but from the right
                    u[i] = self.u0(x0[1])
                elif x0[1] > x0RL: # in shock envelope of the shock on the right, on the left
                    u[i] = self.u0(x0[0])
                else:
                    raise Exception('Something has gone wrong',x0, x0L, x0R)
        else:
            # characteristics haven't crossed yet
            for i in range(len(x)):
                x0 = self.find_x0(x[i],time)
                u[i] = self.u0(x0)
            
        if reshape:
            u = np.reshape(u,orig_shape,'F')

        return u

    def _configure_characteristic_ic_from_pdebase(self):
        '''
        For ICs from PdeBase.set_q0: u0(x)=set_q0(xy=x) (periodic via modx), u0min/max from samples.
        ``*_coarse*``: set ``self.tb = 0`` (see calc_breaking_time / exact_sol).
        '''
        assert self.dim == 1
        qt = self.q0_type.lower()

        def u0_eval(x0):
            x = self.modx(np.asarray(x0, dtype=float))
            return np.asarray(PdeBase.set_q0(self, q0_type=qt, xy=x), dtype=float)

        self.u0 = u0_eval
        self.du0dx = None
        self.x0b = None
        self.xb = None
        self.tb = None
        self._exact_sol_preshock_only = True

        nn = int(getattr(self, 'nn', 0) or 0)
        xs = np.linspace(self.xmin, self.xmax, max(2049, 32 * max(nn, 1)), dtype=float)
        uf = u0_eval(xs)
        self.u0min = float(np.min(uf))
        self.u0max = float(np.max(uf))

        if 'coarse' in qt:
            self.tb = 0.0
    
    def set_q0(self, q0_type=None, xy=None, **kwargs):
        # overwrite base function from PdeBase
        
        if q0_type is None:
            q0_type = self.q0_type
        q0_type = q0_type.lower()

        if xy is None: xy = self.x_elem

        if ('sinwave' in q0_type) and not ('gassner' in q0_type or 'coarse' in q0_type) \
            or ('coswave' in q0_type) and not ('gassner' in q0_type or 'coarse' in q0_type):

            w = 2*np.pi
            # This named case is used by the General Dissipation Burgers
            # experiment and gives exactly u(x,0) = 2 + sin(2*pi*x).
            if 'shift2' in q0_type:
                b = 2.0
            elif 'smallshift' in q0_type:
                b = 1.1 * self.q0_max_q
            elif 'shift' in q0_type:
                b = 1.5 * self.q0_max_q
            else:
                b = 0.0
            if 'sinwave' in q0_type:
                self.u0 = lambda x0 : np.sin(w * ((x0 + self.xmin)/self.dom_len)) * self.q0_max_q + b
                self.du0dx = lambda x0 : (w/self.dom_len) * np.cos(w * ((x0 + self.xmin)/self.dom_len)) * self.q0_max_q
                self.u0min = -self.q0_max_q + b
                self.u0max = self.q0_max_q + b
                self.tb = self.dom_len/w
                self.x0b = 0.5 * self.dom_len + self.xmin
                self.xb = self.eq_x(self.x0b, self.tb)
            elif 'coswave' in q0_type:
                self.u0 = lambda x0 : np.cos(w * ((x0 + self.xmin)/self.dom_len)) * self.q0_max_q + b
                self.du0dx = lambda x0 : -(w/self.dom_len) * np.sin(w * ((x0 + self.xmin)/self.dom_len)) * self.q0_max_q
                self.u0min = -self.q0_max_q + b
                self.u0max = self.q0_max_q + b
                self.tb = self.dom_len/w
                self.x0b = 0.25 * self.dom_len + self.xmin
                self.xb = self.eq_x(self.x0b, self.tb)

            self._exact_sol_preshock_only = False

            if 't05' in q0_type:
                if self.tfix_initial_time is None:
                    self.tfix_initial_time = self.tb*0.5

                if xy.ndim >1: 
                    reshape=True
                    orig_shape = xy.shape
                    xtmp = xy.flatten('F')
                else:
                    xtmp = xy

                q0 = np.empty_like(xtmp) 
                for i in range(len(xtmp)):
                    x0 = self.find_x0(xtmp[i],self.tfix_initial_time)
                    q0[i] = self.u0(x0)# initiate u

                if reshape:
                    q0 = np.reshape(q0,orig_shape,'F')
                self.tb = None
            
            else:
                q0 = self.u0(xy)
        
        else:
            q0 = PdeBase.set_q0(self, q0_type=q0_type, xy=xy)
            if getattr(q0, 'ndim', 0) == 2 and q0.shape == (self.nen, self.nelem):
                q0 = fn.repeat_neq_gv(q0, self.neq_node)
            if q0_type == self.q0_type.lower():
                self._configure_characteristic_ic_from_pdebase()
            elif self.u0min is None:
                if kwargs.get('print_warning', True):
                    print("WARNING: Auxiliary q0_type for this set_q0 call; exact_sol is not "
                          "defined until set_q0 runs with the diffeq primary q0_type.")
                self.u0min = float(np.min(q0))
                self.u0max = float(np.max(q0))
                self.u0 = lambda x0, _qt=q0_type: PdeBase.set_q0(self, q0_type=_qt, xy=x0)
        
        return q0

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
        return np.max(fn.cabs(q))

    def dExdq_abs(self, q, entropy_fix):

        dExdq_abs = fn.gdiag_to_gbdiag(fn.cabs(q))
        return dExdq_abs
    
    def maxeig_dExdq(self, q):
        ''' return the absolute maximum eigenvalue - used for LF fluxes '''
        return fn.cabs(q)
    
    def maxeig_dEndq(self, q, dxidx):
        ''' return the absolute maximum eigenvalue - used for LF fluxes '''
        return fn.cabs(q*dxidx)
    
    def entropy_var(self,q):
        return q
    
    def dqdw(self,q):
        return fn.gdiag_to_gbdiag(np.ones_like(q))
    
    def maxeig_dqdw(self,q):
        return np.ones(q.shape)
    
    def dExdw_abs(self, q, entropy_fix):

        dExdw_abs = fn.gdiag_to_gbdiag(fn.cabs(q))
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
        mn = float(np.min(dqdx))
        assert mn < 0.0, (
            'Burgers calc_breaking_time: the discrete breaking time Tb = -1/min(dqdx) is not defined '
            '(min(dqdx) >= 0). The characteristic / exact-solution framework is not valid for this '
            'discrete initial state in the form expected here.'
        )
        Tb = -1.0 / mn
        qt = self.q0_type.lower()
        coarse_tb0 = 'coarse' in qt and self.tb is not None and np.isfinite(float(self.tb)) and float(self.tb) == 0.0
        if self.tb is not None and np.isfinite(float(self.tb)) and not coarse_tb0:
            assert np.abs(Tb - self.tb) < 1e-2, f'Breaking time {Tb} does not match {self.tb}'
        if self.tb is None and getattr(self, '_exact_sol_preshock_only', False):
            self.tb = Tb
        if print_res:
            print(f'The breaking time is approximately T = {Tb:.{sig_fig}g}')
            if self.tb is not None:
                print(f'The exact breaking time is T = {self.tb:.{sig_fig}g}')
        else:
            # For coarse Gassner IC, self.tb stays 0 for exact_sol / characteristics, but callers
            # (e.g. t_final = calc_breaking_time() * tf) need the discrete gradient scale Tb > 0.
            if coarse_tb0 and getattr(self, '_exact_sol_preshock_only', False):
                return Tb
            return self.tb if self.tb is not None else Tb
        
    def u0(self, x0): raise NotImplementedError('Initial condition not set')
    def du0dx(self, x0): raise NotImplementedError('Initial condition not set')
    
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
    
    def plot_characteristics(self, t, N=30):
        ''' plot the (N) characteristics of the Burgers equation '''
        plt.figure()
        ts = np.linspace(0,t,100)
        for x0 in np.linspace(0,1,N):
            plt.plot(self.modx(self.eq_x(x0,t)),ts,'.')
        plt.plot(self.xb,self.tb,'s',color='blue') # where characteristics first cross
        #x0L, x0R = self.find_envelope(tf)
        #if x0L is not None:
        #    plt.plot(self.eq_x(x0L,t),t,'s',color='purple') # edge of `shocked envelope'
        #    plt.plot(self.eq_x(x0R,t),t,'s',color='purple')
        plt.hlines(self.tb,xmin=0,xmax=1,linestyle='--',color='k') # breaking time
        plt.xlabel('x')
        plt.ylabel('t')
        plt.ylim(0,t)
        plt.xlim(self.xmin,self.xmax)
    
    @staticmethod
    def find_all_roots(eq, args, domain=(0,1), N=30, tol=1e-13):
        """ Find all x0 in range [domain[0],domain[1]] such that eq(x0,t,x)==0."""
        if not isinstance(args, tuple): args = (args,)
        eq_res_accept = 1e-10
        x = np.linspace(domain[0], domain[1], N+1)
        dx = x[1] - x[0]
        f = eq(x, *args)
        df = np.diff(f)
        abs_eq = lambda *z : np.abs(eq(*z))

        roots = []
        for i in range(N):
            x0 = None
            if np.abs(f[i]) < tol: # done, we found a root!
                x0 = x[i]
            elif f[i]*f[i+1] < 0.0: # there is a root between these points
                sol = root_scalar(eq, args=args, bracket=(x[i], x[i+1]), method='brentq', xtol=tol)
                x0 = sol.root
            elif i > 0 and df[i]*df[i-1] < 0.0: # sign change in derivative (skip i==0: df[-1] is meaningless)
                try:
                    # first loosely solve for the minimum, then finely solve if it looks like a root
                    sol = minimize_scalar(abs_eq, args=args, bracket=(x[i]-dx, x[i+1]), method='golden', tol=1e-6)
                    if sol.success: 
                        x0_ = sol.x
                        if abs_eq(x0_,*args) < 1e-6 and not any(abs(x0_ - r) < 1e-3 for r in roots):
                            sol = root_scalar(eq, args=args, bracket=(x0_ - 1e-3, x0_ + 1e-3), method='brentq', xtol=tol)
                            x0 = sol.root
                except:
                    pass
            if x0 is not None:
                if abs_eq(x0,*args) < eq_res_accept and not any(abs(x0 - r) < tol*10 for r in roots): 
                    roots.append(x0)

        return np.sort(roots)
    
    def exact_sol_old(self, time=0, x=None, guess=None):

        if x is None:
            x = self.x_elem
        
        if guess is None:
            assert(guess.shape == x.shape),'guess must be the same shape as x'

        reshape = False
        if x.ndim >1: 
            reshape=True
            orig_shape = x.shape
            x = x.flatten('F')
            if guess is not None:
                guess = guess.flatten('F')

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
                            if f(eps)*f(shock-eps) > 0:
                                # If the function has the same sign at both endpoints, we need to use a different bracket.
                                print('WARNING: The function has the same sign at both endpoints.')
                                print(f(eps), f(shock-eps), xi, time)
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
                #print(f'WARNING: Time {time:.3g} is greater than the breaking time {Tb:.3g}.')
                #print('          Ignoring analytical solution because jump conditions are not yet implemented.')
                #u = np.zeros_like(x)

                for i in range(len(x)):
                    u0 = lambda x0 : self.set_q0(xy=x0)
                    modx = lambda x0: np.mod(x0-self.xmin,self.dom_len) + self.xmin
                    eq = lambda x0 : modx(x[i] - u0(x0)*time) - x0
                    if guess is None:
                        xguess = modx(x[i]-u0(x[i])*time)
                        xguess2 = None
                    else:
                        xguess = modx(x[i]-guess[i]*time)
                        xguess2 = modx(x[i]-u0(x[i])*time)
                    x0 = newton(eq,x0=xguess,x1=xguess2,tol=1e-12,maxiter=1000)
                    u[i] = u0(x0)
            
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
