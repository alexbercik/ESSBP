#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 16:58:18 2020

@author: andremarchildon
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.optimize import newton

from Source.DiffEq.DiffEqBase import PdeBaseCons
from Source.DiffEq.SatBase import SatBaseCons

'''
There are several classes in this file each of them inherit from other classes.
Here is the breakdown of class inheritance:
    Quasi1dEulerExaSol  <- Quasi1dEulerBase
    Quasi1dEulerFd      <- Quasi1dEulerNum <- Quasi1dEulerBase
    Quasi1dEulerSbp     <- SatBaseCons
                        <- Quasi1dEulerNum <- Quasi1dEulerBase

Quasi1dEulerBase:
    -Sets several parameters for this PDE
    -Contains the methods to calculate the shape of the conv. div. nozzle
    -Contains several methods to decompose the flux into its ind. components
    (q_0, q_1, q_2) and conserved variables (rho, u, e)

Quasi1dEulerExaSol:
    -Calculates the exact solution for the conv. div. nozzle

Quasi1dEulerNum:
    -Contains parameters used to solve the PDE
    -Contains an init file to set parameters and the BCs
    -Contains methods to calculate the flux the source term and methods to
    calculate their derivatives wrt the solution

Quasi1dEulerFd:
    -Larger init file to calculate required derivative operators along with BC
    vectors and operators for the artificial dissipation
    -Contains methods for the artificial dissipation
    -Methods for the Riemann invarient BC

Quasi1dEulerSbp:
    -SBP specific methods
'''

class Quasi1dEulerBase(PdeBaseCons):

    # Equation constants
    diffeq_name = 'Quasi1dEuler'
    dim = 1
    neq_node = 3            # No. of equations per node
    npar = 0                # No. of design parameters
    has_exa_sol = True

    # Problem constants
    R = 287
    g = 1.4
    nozzle_shape = 'book' # 'book', 'constant', 'linear'
    xmin_nozzle = 0  # Set based on the nozzle geometry
    xmax_nozzle = 10 # Set based on the nozzle geometry
    isperiodic = False
    length_nozzle = xmax_nozzle - xmin_nozzle

    # Plotting constants
    plt_var2plot_name = 'rho' # rho, u, e, p, a

    def fun_s(self, xvec_in):

        if self.nozzle_shape == 'book':
            return self.fun_s_book(xvec_in)
        elif self.nozzle_shape == 'constant':
            return self.fun_s_const(xvec_in)
        elif self.nozzle_shape == 'linear':
            return self.fun_s_linear(xvec_in)
        else:
            raise Exception('Unknown nozzle shape')

    def fun_der_s(self, xvec_in):

        if self.nozzle_shape == 'book':
            return self.fun_der_s_book(xvec_in)
        elif self.nozzle_shape == 'constant':
            return self.fun_der_s_const(xvec_in)
        elif self.nozzle_shape == 'linear':
            return self.fun_der_s_linear(xvec_in)
        else:
            raise Exception('Unknown nozzle shape')

    def fun_s_book(self, xvec_in):

        if xvec_in.ndim == 1:
            xvec = xvec_in
        elif xvec_in.ndim == 2:
            xvec = xvec_in[:,0]
        else:
            raise Exception('Too many dimensions for xvec')

        def fun_s_part1(x_in):
                return 1 + 1.5*(1-x_in/5)**2

        def fun_s_part2(x_in):
            return 1 + 0.5*(1-x_in/5)**2

        if len(xvec) == 1:
            if xvec < 5:
                return fun_s_part1(xvec)
            else:
                return fun_s_part2(xvec)
        else:
            svec = fun_s_part1(xvec)
            idx = np.argmax(xvec>5)
            svec[idx:] = fun_s_part2(xvec[idx:])

        return svec

    def fun_der_s_book(self, xvec_in):

        if xvec_in.ndim == 1:
            xvec = xvec_in
        elif xvec_in.ndim == 2:
            xvec = xvec_in[:,0]
        else:
            raise Exception('Too many dimensions for xvec')

        svec = -(3/5)*(1-xvec/5)
        idx = np.argmax(xvec>5)
        if len(xvec)>1:
            svec[idx:] = -(1/5)*(1-xvec[idx:]/5)

        return svec

    # Nozzle with constantly changing shape (constant dsdx)
    def fun_s_linear(xvec):
        svec = 2 - xvec/10
        return svec

    def fun_der_s_linear(xvec):
        svec = -np.ones(xvec.shape)/10
        return svec

    # Nozzle of constant shape (dsdx=0)
    def fun_s_const(xvec):
        svec = np.ones(xvec.shape)
        return svec

    def fun_der_s_const(xvec):
        svec = -np.zeros(xvec.shape)
        return svec

    def normalize_q(self, q, q_norm, s_at_q_norm, len_norm):

        rho_norm, _, _, p_norm, a_norm = self.cons2prim(q_norm, s_at_q_norm)

        q_0, q_1, q_2 = np.copy(self.decompose_q(q))

        q_0 /= rho_norm * len_norm
        q_1 /= rho_norm * a_norm * len_norm
        q_2 /= rho_norm * a_norm**2 * len_norm

        q_normalized = self.assemble_vec((q_0, q_1, q_2))
        return q_normalized

    def decompose_q(self, q):

        q_0 = q[0::self.neq_node]
        q_1 = q[1::self.neq_node]
        q_2 = q[2::self.neq_node]

        return q_0, q_1, q_2

    @staticmethod
    def assemble_vec(input_vec):

        nvec = len(input_vec)   # No. of ind. vectors
        n = input_vec[0].size   # Length of each ind. vector
        vec = np.stack(input_vec).reshape(n*nvec, order='f')

        return vec

    @staticmethod
    def prim2cons(rho, u, e, svec):

        q_0 = rho * svec
        q_1 = q_0 * u
        q_2 = e * svec

        q = Quasi1dEulerBase.assemble_vec((q_0, q_1, q_2))

        return q

    def cons2prim(self, q, svec):

        q_0, q_1, q_2 = self.decompose_q(q)

        rho = q_0 / svec
        u = q_1 / q_0
        e = q_2 / svec
        P = (self.g-1)*(e - (rho * u**2)/2)

        a = np.sqrt(self.g * P/rho)

        # Convert numpy array with 1 element to a scalar
        if q.size == 3:
            rho = rho[0]
            u = u[0]
            e = e[0]
            P = P[0]
            a = a[0]
        return rho, u, e, P, a

class Quasi1dEulerExaSol(Quasi1dEulerBase):

    xmin = 0
    xmax = 10
    xthroat = 5
    xshock = 7
    tol=1e-14

    def __init__(self, T_inL, P_inL, s_crit, xy=None,
                 norm_loc='left', fun_s=None, fun_der_s=None):

        if s_crit == 1:
            self.has_shock = True
        else:
            self.has_shock = False

        self.T_inL = T_inL
        self.P_inL = P_inL
        self.s_crit = s_crit

        # self.dim = dim
        self.norm_loc = norm_loc

        # Simple calculations
        self.neq_node = self.dim + 2 # No. of equations at each node
        self.x_len = self.xmax - self.xmin

        # Domain with points on the boundaries
        if xy is None:
            # Only need the exact solution at the start and end of the nozzle
            self.xy = np.array([self.xmin, self.xmax])
        else:
            if xy.ndim == 1:
                self.xy = xy
            elif xy.ndim == 2:
                self.xy = xy[:,0]
            else:
                raise Exception('Size of xy is incompatible')

        self.nn = self.xy.size

        self.svec = self.fun_s(self.xy)
        self.svec_der = self.fun_der_s(self.xy)

        idx = np.argmax(self.xy > self.xshock)
        xvec_1 = np.concatenate((self.xy[:idx], np.array([self.xshock])))
        xvec_2 = np.concatenate((np.array([self.xshock]), self.xy[idx:]))
        self.xy_all = np.concatenate((xvec_1, xvec_2))
        self.svec_all = self.fun_s(self.xy_all)
        self.svec_der_all = self.fun_der_s(self.xy_all)

        self.idx_throat = np.argmax(self.xy_all >= self.xthroat)
        self.idx_shock = np.argmax(self.xy_all >= self.xshock) +1 # +1 since there are 2 nodes here

        # Initiate variables
        self.mach_all = np.zeros(self.nn+2) # +2 for BC and +2 for shock
        self.rho_all = np.zeros(self.nn+2)
        self.u_all = np.zeros(self.nn+2)
        self.e_all = np.zeros(self.nn+2)
        self.P_all = np.zeros(self.nn+2)
        self.T_all = np.zeros(self.nn+2)
        self.a_all = np.zeros(self.nn+2)

        # Solve for the exact solution
        self.calc_exact_sol()

        self.q_all = self.prim2cons(self.rho_all, self.u_all, self.e_all, self.svec_all)

        # Get the solution without the two extra nodes at the shock
        # This allows the solution to be compared with the numerical solution
        def mod_sol(var_all):
            return np.delete(var_all, (self.idx_shock-1, self.idx_shock))

        self.mach = mod_sol(self.mach_all)
        self.rho = mod_sol(self.rho_all)
        self.u = mod_sol(self.u_all)
        self.e = mod_sol(self.e_all)
        self.P = mod_sol(self.P_all)
        self.T = mod_sol(self.T_all)
        self.a = mod_sol(self.a_all)

        self.q = self.prim2cons(self.rho, self.u, self.e, self.svec)

        # Normalize the exact solution
        if self.norm_loc == 'left':
            self.q_ref_norm = self.q[:3]
            self.s_at_q_norm = self.svec[0]
        elif self.norm_loc == 'right':
            self.q_ref_norm = self.q[-3:]
            self.s_at_q_norm = self.svec[-1]
        else:
            raise Exception('Normalization location is not valid')

        self.q_norm = self.normalize_q(self.q, self.q_ref_norm, self.s_at_q_norm, self.x_len)

        # Calculate the variables for the normalized solution
        self.svec_norm = self.svec/self.x_len
        self.xy_norm = self.xy / self.x_len
        self.rho_norm, self.u_norm, self.e_norm, self.P_norm, self.a_norm \
            = self.cons2prim(self.q_norm, self.svec_norm)

    def solve_mach(self, mach_in, svec, s_crit):

        def fun_mach(mach_in):

            k = (2/(self.g+1)) * (1 + 0.5*(self.g-1) * mach_in**2)
            exp = (self.g+1)/(2*(self.g-1))

            return k**exp /mach_in - svec / s_crit

        def der_fun_mach(mach_in):

            k = (2/(self.g+1)) * (1 + 0.5*(self.g-1) * mach_in**2)
            exp = (self.g+1)/(2*(self.g-1))

            return k**exp * (-1/mach_in**2 + 1/k)

        mach_calc = newton(fun_mach, mach_in, fprime=der_fun_mach, tol=self.tol, maxiter=100)

        return mach_calc

    def calc_exact_sol(self):

        def calc_T(T_in, mach_in):
            k = 1 + (self.g-1)/2 * mach_in**2
            T = T_in / k
            return T

        def calc_p(p_in, mach_in):
            k = 1 + (self.g-1)/2 * mach_in**2
            p = p_in * k**(-self.g/(self.g-1))
            return p

        def RankineHugoniot(machL, s_critL):

            # Calcualte values across the shock
            T0R = self.T_inL

            k1 = 0.5*(self.g+1)*machL**2
            k2 = 1 + 0.5*(self.g-1)*machL**2
            k3 = (k1/k2)**(self.g/(self.g-1))
            k4 = ( (2*self.g/(self.g+1)) * machL**2 - (self.g-1)/(self.g+1))**(1/(self.g-1))
            P0R = self.P_inL * k3/k4

            rho0R = P0R / (self.R * T0R)
            a0R = np.sqrt(self.g * P0R / rho0R)
            rho_a_R_star = rho0R * a0R * (2/(self.g+1))**((self.g+1)/(2*(self.g-1)))

            p01 = self.P_inL
            T01 = self.T_inL
            rho01 = p01 / (self.R * T01)
            a01 = np.sqrt(self.g*p01 / rho01)
            rho_a_L_star = rho01*a01*(2/(self.g+1))**((self.g+1)/(2*(self.g-1)))

            s_crit = s_critL * rho_a_L_star / rho_a_R_star

            return s_crit, T0R, P0R

        if self.has_shock:
            # Calculate the mach number up to the throat
            mach_in = np.linspace(0.25, 0.95, self.idx_throat)
            svec_pre_throat = self.svec_all[:self.idx_throat]
            mach_pre_throat = self.solve_mach(mach_in, svec_pre_throat, self.s_crit)
            self.mach_all[:self.idx_throat] = mach_pre_throat

            # Calculate the mach number from the throat to the shock
            n_idx = self.idx_shock - self.idx_throat
            mach_in = np.linspace(1.01, 1.2, n_idx)
            svec_pre_shock = self.svec_all[self.idx_throat:self.idx_shock]
            mach_pre_shock = self.solve_mach(mach_in, svec_pre_shock, self.s_crit)
            self.mach_all[self.idx_throat:self.idx_shock] = mach_pre_shock

            self.T_all[:self.idx_shock] = calc_T(self.T_inL, self.mach_all[:self.idx_shock])
            self.P_all[:self.idx_shock] = calc_p(self.P_inL, self.mach_all[:self.idx_shock])

             # Calculate the variables after the shock
            mach_L_shock = self.mach_all[self.idx_shock-1]
            s_crit_2, T0R, p0R = RankineHugoniot(mach_L_shock, self.s_crit)

            # Calculate the mach number after the shock
            svec_post_shock = self.svec_all[self.idx_shock:]
            mach_in = np.linspace(0.75, 0.4, svec_post_shock.size)
            mach_post_shock = self.solve_mach(mach_in, svec_post_shock, s_crit_2)
            self.mach_all[self.idx_shock:] = mach_post_shock

            self.T_all[self.idx_shock:] = calc_T(T0R, mach_post_shock)
            self.P_all[self.idx_shock:] = calc_p(p0R, mach_post_shock)

        else:
            mach_in = 0.2*np.ones(self.mach_all.shape)
            self.mach_all = self.solve_mach(mach_in, self.svec_all, self.s_crit)
            self.T_all = calc_T(self.T_inL, self.mach_all)
            self.P_all = calc_p(self.P_inL, self.mach_all)

        # Calculate the rest of the parameters
        self.rho_all = self.P_all / (self.R * self.T_all)
        self.a_all = np.sqrt(self.g * self.R * self.T_all)
        self.u_all = self.a_all * self.mach_all
        self.e_all = self.P_all/(self.g-1) + self.rho_all * self.u_all**2 /2

class Quasi1dEulerNum(Quasi1dEulerBase):

    # Constant parameters
    T0 = 300        # Degrees Kelvin
    P0 = 100*1000   # Pressure in Pa

    # Initial and boundary conditions
    init_sol_type = 'linear'
    bc_type = 'dirichlet'     # 'dirichlet', 'riemann'

    # Parameters for the solvers
    use_local_dt = False

    # Subsonic and supersonic parameters
    sc_subsonic = 0.8
    k2_subsonic = 0
    k4_subsonic = 1/50

    sc_supersonic = 1
    k2_supersonic = 1/2
    k4_supersonic = 1/50

    # Normalizing the solution
    norm_var = True
    norm_loc = 'left'

    # Plotting
    calc_exa_sol = True

    def __init__(self, para=None, obj_name=None, q0_type=None, flow_is_subsonic=True):

        ''' Add inputs to the class '''

        Quasi1dEulerBase.__init__(self, para, obj_name, q0_type)
        self.flow_is_subsonic = flow_is_subsonic

        ''' Set flow dependent parameters  '''

        if self.flow_is_subsonic:
            self.sc = self.sc_subsonic
            self.k2 = self.k2_subsonic
            self.k4 = self.k4_subsonic
        else:
            self.sc = self.sc_supersonic
            self.k2 = self.k2_supersonic
            self.k4 = self.k4_supersonic

        ''' Calculate the exact solution at the boundary to apply the BC '''

        q1d_exa = Quasi1dEulerExaSol(self.T0, self.P0, self.sc)
        self.qL_in = q1d_exa.q[:3]
        self.qR_in = q1d_exa.q[-3:]

        ''' Parameters at the boundaries '''

        self.s_at_qL = self.fun_s(np.array([self.xmin_nozzle]))
        self.s_at_qR = self.fun_s(np.array([self.xmax_nozzle]))

        if self.norm_loc == 'left':
            self.q_ref_norm = self.qL_in
            self.s_at_q_norm = self.s_at_qL
        elif self.norm_loc == 'right':
            self.q_ref_norm = self.qR_in
            self.s_at_q_norm = self.s_at_qR
        else:
            raise Exception('Normalization location is not valid')

        if self.norm_var:
            self.len_norm = self.length_nozzle
            self.qL = self.normalize_q(self.qL_in, self.q_ref_norm, self.s_at_q_norm, self.len_norm)
            self.qR = self.normalize_q(self.qR_in, self.q_ref_norm, self.s_at_q_norm, self.len_norm)

            self.s_at_qL /= self.len_norm
            self.s_at_qR /= self.len_norm
        else:
            self.qL = self.qL_in
            self.qR = self.qR_in

        self.rhoL, self.uL, self.eL, self.PL, self.aL = self.cons2prim(self.qL, self.s_at_qL)
        self.rhoR, self.uR, self.eR, self.PR, self.aR = self.cons2prim(self.qR, self.s_at_qR)

        self.EL = self.calcE(self.qL)
        self.ER = self.calcE(self.qR)

    def exact_sol(self, _):
        return self.q_exa

    def set_mesh(self, mesh):

        Quasi1dEulerBase.set_mesh(self, mesh)

        if self.norm_var:
            assert self.xmin == 0 and self.xmax == 1, \
                'For solve with normalized solution must have xmin=0 and xmax=1'
            self.xy_not_norm = self.xy * self.length_nozzle
        else:
            assert self.xmin == self.xmin_nozzle and self.xmax == self.xmax_nozzle, \
                'For solve without normalized solution must have xmin=xmin_nozzle and xmax=xmax_nozzle'
            self.xy_not_norm = self.xy

        # Calculate the shape of the nozzle at the mesh and boundary nodes
        self.svec = self.fun_s(self.xy_not_norm)
        self.svec_der = self.fun_der_s(self.xy_not_norm)

        if self.norm_var:
            self.svec /= self.length_nozzle

    def set_q0(self):

        if self.calc_exa_sol or self.init_sol_type == 'exact':
            self.exa_sol = Quasi1dEulerExaSol(self.T0, self.P0, self.sc, self.xy_not_norm)

            if self.norm_var:
                self.q_exa = self.exa_sol.q_norm
            else:
                self.q_exa = self.exa_sol.q

        if self.init_sol_type == 'exact':

            if self.norm_var:
                self.q_init = self.exa_sol.q_norm
            else:
                self.q_init = self.exa_sol.q

        elif self.init_sol_type == 'linear':

            rho_linear = np.linspace(self.rhoL, self.rhoR, self.nn)
            u_linear = np.linspace(self.uL, self.uR, self.nn)
            e_linear = np.linspace(self.eL, self.eR, self.nn)

            self.q_init = self.prim2cons(rho_linear, u_linear, e_linear, self.svec)
        else:
            raise Exception('Unknown init sol type')

        # Calculate the primative and conserved variables for the init solution
        self.q = np.copy(self.q_init)
        self.q_0, self.q_1, self.q_2 = self.decompose_q(self.q)

        self.rho_init, self.u_init, self.e_init, self.P_init, self.a_init = self.cons2prim(self.q, self.svec)
        self.rho, self.u, self.e, self.P, self.a = self.cons2prim(self.q, self.svec)

        return self.q_init

    def var2plot(self, q_in):

        rho, u, e, P, a = self.cons2prim(q_in, self.svec)

        if self.plt_var2plot_name == 'rho':
            return rho
        elif self.plt_var2plot_name == 'u':
            return u
        elif self.plt_var2plot_name == 'e':
            return e
        elif self.plt_var2plot_name == 'p':
            return P
        elif self.plt_var2plot_name == 'a':
            return a
        else:
            raise Exception('Requested variable to plot is not available')

    def calcE(self, q=None):

        if q is None:
            u = self.u
            q_1 = self.q_1
            q_2 = self.q_2
        else:
            q_0, q_1, q_2 = self.decompose_q(q)
            u = q_1 / q_0

        k = u*q_1                    # Common term: rho * u^2 * S
        ps = (self.g-1)*(q_2 - k/2)  # p * S

        e0 = q_1
        e1 = k + ps
        e2 = u*(q_2 + ps)

        E = self.assemble_vec((e0, e1, e2))
        return E

    def dEdq(self, q=None):

        if q is None:
            u = self.u
            q_0 = self.q_0
            q_2 = self.q_2
        else:
            q_0, q_1, q_2 = self.decompose_q(q)
            u = q_1 / q_0

        n_node = q_0.shape[0]

        # dEdq is complex if using complex step for implicit time
        # marching with SBP operators
        if (q is not None) and any(np.iscomplex(q)):
            dEdq = np.zeros((n_node, 3, 3), dtype=complex)
        else:
            dEdq = np.zeros((n_node, 3, 3))

        u2 = u**2
        q2_q0 = q_2/q_0

        dEdq[:, 0, 1] = 1
        dEdq[:, 1, 0] = 0.5*(self.g-3) * u2
        dEdq[:, 1, 1] = (3-self.g) * u
        dEdq[:, 1, 2] = self.g-1
        dEdq[:, 2, 0] = (self.g-1) * (u * u2) - self.g * q2_q0 * u
        dEdq[:, 2, 1] = self.g * q2_q0 - (3/2)*(self.g-1) * u2
        dEdq[:, 2, 2] = self.g * u

        if n_node == 1:
            dEdq = dEdq[0,:,:]
        else:
            dEdq = sp.block_diag(dEdq)

        return dEdq

    def calcG(self, q=None, xy_idx0=None, xy_idx1=None):

        if q is None:
            q_0 = self.q_0
            q_1 = self.q_1
            q_2 = self.q_2
            svec = self.svec
            svec_der = self.svec_der
        else:
            # These lines of code are used when G is calculated for individual
            # elements. The inputs xy_idx0 and xy_idx1 indicate the index of
            # the nodes for the element with solution q
            q_0, q_1, q_2 = self.decompose_q(q)
            svec = self.svec[xy_idx0:xy_idx1]
            svec_der = self.svec_der[xy_idx0:xy_idx1]

        p = (self.g-1)*(q_2 - 0.5*q_1**2 /q_0) /svec

        g2 = p * svec_der

        zero_vec = np.zeros(g2.shape[0])
        G = self.assemble_vec((zero_vec, g2, zero_vec))
        return G

    def dGdq(self, q=None, xy_idx0=None, xy_idx1=None):

        if q is None:
            u = self.u
            svec = self.svec
            svec_der = self.svec_der
        else:
            # These lines of code are used when G is calculated for individual
            # elements. The inputs xy_idx0 and xy_idx1 indicate the index of
            # the nodes for the element with solution q
            q_0, q_1, q_2 = self.decompose_q(q)
            u = q_1 / q_0
            svec = self.svec[xy_idx0:xy_idx1]
            svec_der = self.svec_der[xy_idx0:xy_idx1]

        u2 = u**2
        k = svec_der / svec # Common term

        n_node = u.shape[0]

        # dGdq is complex if using complex step for implicit time
        # marching with SBP operators
        if (q is not None) and any(np.iscomplex(q)):
            dGdq = np.zeros((n_node, 3, 3), dtype=complex)
        else:
            dGdq = np.zeros((n_node, 3, 3))

        dGdq[:, 1, 0] = 0.5*(self.g-1)*u2 * k
        dGdq[:, 1, 1] = -(self.g-1)*u * k
        dGdq[:, 1, 2] = self.g-1 * k

        if n_node == 1:
            dGdq = dGdq[0,:,:]
        else:
            dGdq = sp.block_diag(dGdq)

        return dGdq

class Quasi1dEulerFd(Quasi1dEulerNum):

    def set_fd_op(self, p, use_sparse):

        Quasi1dEulerNum.set_fd_op(self, p, use_sparse)

        # Derivative operators and boundary vectors for the artificial dissipation
        self.op_del, self.op_inv_del, self.op_del3, self.ad_bc_2L, \
            self.ad_bc_2R, self.ad_bc_4L, self.ad_bc_4R = self.makeAdOp()

    def dqdt(self, q):

        # Update the primative and conserved variables of the solution
        self.q = q
        self.rho, self.u, self.e, self.P, self.a = self.cons2prim(q, self.svec)
        self.q_0, self.q_1, self.q_2 = self.decompose_q(q)

        # if self.bc_type == 'riemann':
        self.bc_riemann()

        # dEdx = self.dEdx(q)
        # G = self.calcG()
        # L, D = self.artificial_dissipation()

        dEdx = self.dEdx(q)
        G = self.calcG(q)
        _, D = self.artificial_dissipation(q)

        dqdt = -dEdx + G + D
        # dqdt = -dEdx + G

        return dqdt

    def dfdq(self, q):

        # Update the primative and conserved variables of the solution
        # self.rho, self.u, self.e, self.P, self.a = self.cons2prim(q, self.svec)
        # self.q_0, self.q_1, self.q_2 = self.decompose_q(q)

        # A = self.dEdq()
        # dAdx = self.dAdx(A)
        # dGdq = self.dGdq()
        # L, D = self.artificial_dissipation()

        A = self.dEdq(q)
        dAdx = self.dAdx(A)
        dGdq = self.dGdq(q)
        L, _ = self.artificial_dissipation(q)

        dfdq = (-dAdx + dGdq + L)
        # dfdq = -dAdx + dGdq

        return dfdq

    def makeAdOp(self):
        # Makes the operators for the artificial dissipation

        ''' Derivative operators '''

        # Create the matrix operator inv_del of size [3n x 3(n+1)]
        v_j = -np.ones(self.nn)
        v_jp1 = np.ones(self.nn)
        op_inv_del = sp.diags((v_j, v_jp1), (0, 1), shape=(self.nn, self.nn+1))
        op_inv_del = sp.kron(op_inv_del, sp.eye(self.neq_node))

        # Create the matrix operator del of size [3(n+1) x 3n]
        v_jn1 = -np.ones(self.nn)
        v_j = np.ones(self.nn)
        op_del = sp.diags((v_jn1, v_j), (-1,0), shape=(self.nn+1, self.nn))
        op_del = sp.kron(op_del, sp.eye(self.neq_node))

        # Create the matrix operator (del @ inv_del @ del) of size [3(n+1) x 3n]
        v_0 = -3*np.ones(self.nn)
        v_0[0] = -2
        v_n1 = 3*np.ones(self.nn)
        v_n1[-1] = 2
        v_p1 = np.ones(self.nn-1)
        v_n2 = -np.ones(self.nn-1)
        op_del3 = sp.diags((v_n2, v_n1, v_0, v_p1), (-2, -1, 0, 1), shape=(self.nn+1, self.nn))
        op_del3 = sp.kron(op_del3, sp.eye(self.neq_node))

        ''' Vetors to apply boundary conditions for the artificial dissipation'''

        # 2nd order left side
        data = [-1]
        location = ([0], # row
                    [0]) # col
        ad_bc_2L = sp.coo_matrix((data, location), shape=(self.nn+1, 1))
        ad_bc_2L = sp.kron(ad_bc_2L, sp.eye(3))

        # 2nd order right side
        data = [1]
        location = ([self.nn],   # row
                    [0])        # col
        ad_bc_2R = sp.coo_matrix((data, location), shape=(self.nn+1, 1))
        ad_bc_2R = sp.kron(ad_bc_2R, sp.eye(3))

        # 4th order left side
        data = [1, -1]
        location = ([0, 1], # row
                    [0, 0]) # col
        ad_bc_4L = sp.coo_matrix((data, location), shape=(self.nn+1, 1))
        ad_bc_4L = sp.kron(ad_bc_4L, sp.eye(3))

        # 4th order right side
        data = [1, -1]
        location = ([self.nn-1, self.nn], # row
                    [0, 0])             # col
        ad_bc_4R = sp.coo_matrix((data, location), shape=(self.nn+1, 1))
        ad_bc_4R = sp.kron(ad_bc_4R, sp.eye(3))

        return op_del, op_inv_del, op_del3, ad_bc_2L, ad_bc_2R, ad_bc_4L, ad_bc_4R

    def dEdx(self, q):

        E = self.calcE(q)

        dEdx = self.der1 @ E
        dEdx[:self.neq_node] -= self.EL / (2*self.dx)
        dEdx[-self.neq_node:] += self.ER / (2*self.dx)

        return dEdx

    def dAdx(self, A):

        dAdx = self.der1 @ A

        return dAdx

    def artificial_dissipation(self, q_in=None):

        if q_in is None:
            q = self.q
            u = self.u
            P = self.P
            a = self.a
        else:
            q = q_in
            rho, u, e, P, a = self.cons2prim(q, self.svec)

        # Calculate the pressure sensor
        p_mod = np.concatenate((np.array([self.PL]), P, np.array([self.PR]) ))
        gamma = abs((p_mod[2:] - 2*p_mod[1:-1] + p_mod[:-2])/ (p_mod[2:] + 2*p_mod[1:-1] + p_mod[:-2]))

        # TODO: add option for matrix dissipation
        eps2 = np.zeros(self.nn+2)
        eps2[0] = self.k2 * np.max(gamma[:2])
        eps2[1:-1] = self.k2 * np.max((gamma[2:], gamma[1:-1], gamma[0:-2]))
        eps2[-1] = self.k2 * np.max(gamma[-2:])
        # eps2 = self.assemble_vec((eps2, eps2, eps2))

        eps4 = np.max((np.zeros(eps2.shape), self.k4-eps2), axis=0)

        # Calculate the maximum eigenvalue
        max_eig = np.zeros(self.nn+2)
        max_eig[1:-1] = (np.abs(u) + a)
        max_eig[0] = (np.abs(self.uL) + self.aL)
        max_eig[-1] = (np.abs(self.uR) + self.aR)

        # TODO: add option for Roe average
        max_eig_eps2 = eps2 * max_eig
        max_eig_eps4 = eps4 * max_eig

        L2_temp = 0.5*(max_eig_eps2[:-1] + max_eig_eps2[1:])
        L4_temp = 0.5*(max_eig_eps4[:-1] + max_eig_eps4[1:])

        L2_part_1 = self.op_inv_del @ sp.diags(self.assemble_vec((L2_temp, L2_temp, L2_temp)) ) / self.dx
        L4_part_1 = self.op_inv_del @ sp.diags(self.assemble_vec((L4_temp, L4_temp, L4_temp)) ) / self.dx

        L2 = L2_part_1 @ self.op_del
        L4 = L4_part_1 @ self.op_del3
        L = L2 - L4

        D2 = L2_part_1 @ (self.op_del @ q + self.ad_bc_2L @ self.qL + self.ad_bc_2R @ self.qR)
        D4 = L4_part_1 @ (self.op_del3 @ q + self.ad_bc_4L @ self.qL + self.ad_bc_4R @ self.qR)
        D = D2 - D4

        return L, D

    def solve(self, nts, tm_method, cn):

        # Time step
        if self.use_local_dt:
            self.local_dt(cn)
        else:
            self.dt = sp.eye(3*self.nn) * (cn * self.dx / (np.abs(self.uL) + self.aL))

        self.dt = 0.1 * self.dt / self.dt.todense()[0,0]

        q = self.set_q0()

        res_norm = np.zeros(nts)
        I = sp.eye(self.nn*3)

        for i_ts in range(nts):

            if self.bc_type == 'riemann':
                self.bc_riemann()

            if self.use_local_dt:
                self.local_dt(cn)

            dEdx = self.dEdx(q)
            G = self.calcG()
            L, D = self.artificial_dissipation()

            if tm_method == 'implicit_euler':
                A = self.dEdq()
                dAdx = self.dAdx(A)
                dGdq = self.dGdq()

                ff = -dEdx + G + D
                dfdq = -dAdx + dGdq + L

                LHS = I - self.dt @ dfdq
                RHS = self.dt @ ff

                dq = spsolve(LHS, RHS)

            elif tm_method == 'explicit_euler':
                RHS = -dEdx + G + D
                dq = self.dt @ RHS
            else:
                raise Exception('TM method not available')

            self.q += dq

            # Calculate the norm of the residual
            res_norm[i_ts] = np.linalg.norm(RHS)
            if i_ts % 5 == 0:
                print(f'i = {i_ts:3}, res_norm = {res_norm[i_ts]:.2e}')

            # Update the primative and conserved variables of the solution
            self.rho, self.u, self.e, self.P, self.a = self.cons2prim(self.q, self.svec)
            self.q_0, self.q_1, self.q_2 = self.decompose_q(self.q)

        return self.q, res_norm

    def local_dt(self, cn):

        u = self.u
        a = self.a

        dt = (cn * self.dx) / (np.abs(u) + a)
        self.dt = sp.diags(self.assemble_vec((dt, dt, dt)))

    def bc_riemann(self):

        def char_speeds(vec_out, u, a):
            # vec_out is the outward unit normal vector
            char_speeds = np.array([vec_out*u - a,
                                vec_out*u + a,
                                vec_out*u])

            return char_speeds

        def var2RiemannInv(vec_out, u, a, p, rho):

            reiman_vec = np.array([vec_out*u - 2*a/(self.g-1),
                                   vec_out*u + 2*a/(self.g-1),
                                   np.log(p/(rho**self.g)) ])

            return reiman_vec

        def RiemannInv2var(vec_out, RiemInv):
            u = (RiemInv[0] + RiemInv[1]) / (2*vec_out)
            a = 0.25*(self.g-1)*(RiemInv[1] - RiemInv[0])
            rho = (a**2/self.g * np.e**(-RiemInv[2]))**(1/(self.g-1))
            p = a**2 * rho / self.g
            e = p/(self.g-1) + 0.5*rho*u**2

            return rho, u, e, p, a

        vec_L = -1 # Normal vector for the left boundary
        vec_R = 1 # Normal vector for the right boundary

        # Calculate the characteristic speeds
        char_vL = char_speeds(vec_L, self.uL, self.aL)
        char_vR = char_speeds(vec_R, self.uR, self.aR)

        # Calculate Rieman invarients at the left and right boundaries
        RiemannInvL = var2RiemannInv(vec_L, self.uL, self.aL, self.PL, self.rhoL)
        RiemannInvR = var2RiemannInv(vec_R, self.uR, self.aR, self.PR, self.rhoR)

        # Calculate Rieman invarients at the first and last nodes
        rho_xmin, u_xmin, _, p_xmin, a_xmin = self.cons2prim(self.q[:3], self.svec[0])
        RiemannInv_xmin = var2RiemannInv(vec_L, u_xmin, a_xmin, p_xmin, rho_xmin)
        RiemannInv_Lnew = np.zeros(3)

        rho_xmax, u_xmax, _, p_xmax, a_xmax = self.cons2prim(self.q[-3:], self.svec[-1])
        RiemannInv_xmax = var2RiemannInv(vec_R, u_xmax, a_xmax, p_xmax, rho_xmax)
        RiemannInv_Rnew = np.zeros(3)

        # Set the Riemann invariants at the boundaries
        for i in range(3):
            if char_vL[i] > 0: # Info is leaving the domain
                RiemannInv_Lnew[i] = RiemannInv_xmin[i]
            else:
                RiemannInv_Lnew[i] = RiemannInvL[i]

            if char_vR[i] > 0: # Info is leaving the domain
                RiemannInv_Rnew[i] = RiemannInv_xmax[i]
            else:
                RiemannInv_Rnew[i] = RiemannInvR[i]

        # Update boundary values
        self.rhoL, self.uL, self.eL, self.PL, self.aL = RiemannInv2var(vec_L, RiemannInv_Lnew)
        self.rhoR, self.uR, self.eR, self.PR, self.aR = RiemannInv2var(vec_R, RiemannInv_Rnew)

        # Calculate boundary terms
        self.EL = self.calcE(self.qL)
        self.ER = self.calcE(self.qR)

class Quasi1dEulerSbp(SatBaseCons, Quasi1dEulerNum):

    def dEdq_eig_abs(self, dEdq):

        eig_val, eig_vec = np.linalg.eig(dEdq)
        dEdq_eig_abs = eig_vec @ np.diag(np.abs(eig_val)) @ np.linalg.inv(eig_vec)

        return dEdq_eig_abs

    def dfdq(self, q,  xy_idx0=None, xy_idx1=None):
        A = self.dEdq(q)
        dGdq = self.dGdq(q, xy_idx0, xy_idx1)

        dfdq = -self.der1 @ A + dGdq
        return dfdq