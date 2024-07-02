#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 23:51:27 2021

@author: bercik
"""

import numpy as np
from numba import jit
import scipy.sparse as sp
#from scipy.sparse.linalg import spsolve
from scipy.optimize import newton, bisect

from Source.DiffEq.DiffEqBase import PdeBase
import Source.Methods.Functions as fn
import Source.DiffEq.EulerFunctions as efn
import Source.DiffEq.EulerFunctions_cmplx as ecfn

class Quasi1dEuler(PdeBase):
    
    # Equation constants
    diffeq_name = 'Quasi1dEuler'
    dim = 1
    neq_node = 3            # No. of equations per node
    pde_order = 1
    has_exa_sol = True
    para_names = (r'$R$',r'$\gamma$',)
    nondimensionalize = False
    t_scale = 1.
    a_inf = 1.
    rho_inf = 1.

    # Problem constants
    R_fix = 287
    g_fix = 1.4
    para_fix = [R_fix,g_fix]

    # Plotting constants
    plt_var2plot_name = r'$\rho$' # rho, u, e, p, a, mach    

    def __init__(self, para, q0_type=None, test_case='subsonic_nozzle',
                 nozzle_shape='book', bc='periodic'):

        ''' Add inputs to the class '''

        super().__init__(para, q0_type)
        self.R = para[0]
        self.g = para[1]
        self.test_case = test_case
        self.nozzle_shape = nozzle_shape
        
        if self.q0_type == None:
            self.q0_type = 'linear' # can be exact, linear

        if self.nondimensionalize:
            print('Using non-dimensionalized variables.')

        ''' Set flow and problem dependent parameters  '''
        
        if self.test_case == 'subsonic_nozzle':
            self.xmin_fix = 0  # should be the same as self.x_min
            self.xmax_fix = 10 # should be the same as self.x_max
            self.T01 = 300         # Temperature in Kelvin at inlet
            self.p01 = 100*1000   # Total Pressure in Pa at inlet
            self.s_crit = 0.8     # critical nozzle area. mach=1 when svec=s_crit
            #self.k2 = 0.          # coefficient for first order FD dissipation, usually 0
            #self.k4 = 0.02        # coefficient for third order FD dissipation, usually 0.02
            assert (bc == 'dirichlet' or bc == 'riemann'),\
                "subsonic_nozzle must use bc='dirichlet' or bc='riemann'."
            self.steady = True
            self.t_final = 'steady'

            if self.nondimensionalize:
                # hard-code in exact solution at LHS boundary
                self.rho_inf = 2.852280050286349e+00
                rhou_inf = 1.866846848484981e+02
                e_inf = 6.156988102623681e+05
                self.a_inf = self.calc_a(self.rho_inf, rhou_inf, e_inf)
                self.t_scale = 1.0 # no need since it is steady

            
        elif self.test_case == 'transonic_nozzle':
            self.xmin_fix = 0  # should be the same as self.x_min
            self.xmax_fix = 10 # should be the same as self.x_max
            self.T01 = 300         # Temperature in Kelvin at inlet
            self.p01 = 100*1000   # Total Pressure in Pa at inlet
            self.s_crit = 1.0     # critical nozzle area. mach=1 when svec=s_crit
            self.shockx = 7.      # x location of shock  *not valid for all nozzle shapes*
            self.transx = 5.      # x location of transition, svec=s_crit *not valid for all nozzle shapes*
            #self.k2 = 0.5         # coefficient for first order dissipation, usually 0.5
            #self.k4 = 0.02        # coefficient for third order dissipation, usually 0.02
            assert (bc == 'dirichlet' or bc == 'riemann'),\
                "transonic_nozzle must use bc='dirichlet' or bc='riemann'."
            self.steady = True
            self.t_final = 'steady'

            if self.nondimensionalize:
                # TODO
                # hard-code in exact solution at LHS boundary
                self.rho_inf = 1.
                rhou_inf = 1.
                e_inf = 1.
                self.a_inf = self.calc_a(self.rho_inf, rhou_inf, e_inf)
                self.t_scale = 1.0 # no need since it is steady
            
        elif self.test_case == 'shock_tube':
            self.xmin_fix = 0  # should be the same as self.x_min
            self.xmax_fix = 10 # should be the same as self.x_max
            self.xmembrane = 5    # membrane x point - for shock tube
            self.pL = 1e5         # initial pressire in Pa of left state
            self.rhoL = 1         # initial density in Kg/m^3 of left state
            self.pR = 1e4         # initial pressire in Pa of right state
            self.rhoR = 0.125     # initial density in Kg/m^3 of right state
            self.t_final = 0.0061 # final time to run to, usually 0.0061. Set t_final=None to use this default.
            #self.k2 = 0.5         # coefficient for first order FD dissipation, usually 0.5
            #self.k4 = 0.02        # coefficient for third order FD dissipation, usually 0.02
            if self.nozzle_shape != 'constant':
                print("WARNING: Overwriting inputted nozzle_shape to 'constant'")
                self.nozzle_shape = 'constant'
            if self.q0_type != 'shock_tube':
                print("WARNING: Overwriting inputted q0_type to 'shock_tube'")
                self.q0_type = 'shock_tube'
            assert (bc == 'dirichlet' or bc == 'riemann'),\
                "shock_tube must use bc='dirichlet' or bc='riemann'."
            self.steady = False

            if self.nondimensionalize:
                self.rho_inf = self.rhoL
                p_inf = self.pL
                self.a_inf = np.sqrt(self.g*p_inf/self.rho_inf)
                self.t_scale = self.a_inf
            
        elif self.test_case == 'density_wave':
            self.xmin_fix = -1.  # should be the same as self.x_min
            self.xmax_fix = 1. # should be the same as self.x_max
            self.u0 = 0.1        # initial (ideally constant) velocity
            self.p0 = 20          # initial (ideally constant) pressure
            if self.nozzle_shape != 'constant':
                print("WARNING: Overwriting inputted nozzle_shape to 'constant'.")
                self.nozzle_shape = 'constant'
            if self.q0_type != 'density_wave':
                print("WARNING: Overwriting inputted q0_type to 'density_wave'.")
                self.q0_type = 'density_wave'
            assert (bc == 'periodic'),\
                "density_wave must use bc='periodic'."
            self.steady = False

            if self.nondimensionalize:
                self.rho_inf = 1.
                p_inf = self.p0
                self.a_inf = np.sqrt(self.g*p_inf/self.rho_inf)
                self.t_scale = self.a_inf

        elif self.test_case == 'manufactured_soln':
            self.xmin_fix = 0.  # should be the same as self.x_min
            self.xmax_fix = 2. # should be the same as self.x_max
            if self.nozzle_shape != 'constant':
                print("WARNING: Overwriting inputted nozzle_shape to 'constant'.")
                self.nozzle_shape = 'constant'
            if self.q0_type != 'manufactured_soln':
                print("WARNING: Overwriting inputted q0_type to 'manufactured_soln'.")
                self.q0_type = 'manufactured_soln'
            assert (bc == 'periodic'),\
                "manufactured_soln must use bc='periodic'."
            self.steady = False
            self.calcG = self.calcG_manufactured

            if self.nondimensionalize:
                # exact solution rho = 2 + 0.1*sin(pi*(x-t)), u = 1, e = rho**2
                self.rho_inf = 2.
                p_inf = (self.g-1)*(self.rho_inf**2 - 0.5*self.rho_inf)
                self.a_inf = np.sqrt(self.g*p_inf/self.rho_inf)
                self.t_scale = self.a_inf



            
        else: raise Exception("Test case not understood. Try 'subsonic_nozzle', 'transonic_nozzle', 'shock_tube', or 'density_wave'.")
        
        if self.g == self.g_fix:
            print('Using the fixed g={0} diffeq functions since params match.'.format(self.g_fix))
            self.calcEx = efn.calcEx_1D
            self.dExdq = efn.dExdq_1D
            self.dEndq_eig_abs_dq = efn.dEndq_eig_abs_dq_1D
            self.central_Ex = efn.Central_flux_1D
            self.ismail_roe_Ex = efn.Ismail_Roe_flux_1D
            self.ranocha_Ex = efn.Ranocha_flux_1D
            self.maxeig_dExdq = efn.maxeig_dExdq_1D
            self.maxeig_dExdq_cmplx = ecfn.maxeig_dExdq_1D
            self.maxeig_dEndq = efn.maxeig_dEndq_1D
            self.entropy = efn.entropy_1D
            self.entropy_var = efn.entropy_var_1D
            self.dqdw = efn.symmetrizer_1D
            self.dEndw_abs = efn.dEndw_abs_1D

        if bc != 'periodic':
            
            ''' Calculate the exact solution at the boundary to apply the BC '''
            xx_temp = np.linspace(self.xmin_fix, self.xmax_fix, num=20,endpoint=True)
            q_exa_bdy = self.exact_sol(x=xx_temp)
            self.qL = q_exa_bdy[:3]
            np.set_printoptions(precision=15)
            self.qR = q_exa_bdy[-3:]
    
            ''' Parameters at the boundaries '''   
            self.s_at_qL = self.fun_s(np.array([self.xmin_fix]))[0]
            self.s_at_qR = self.fun_s(np.array([self.xmax_fix]))[0]
                
            self.rhoL, self.uL, self.eL, self.PL, self.aL = self.cons2prim(self.qL, self.s_at_qL)
            self.rhoR, self.uR, self.eR, self.PR, self.aR = self.cons2prim(self.qR, self.s_at_qR)
            
            self.EL = self.calcEx(np.reshape(q_exa_bdy[:3],(3,1))).flatten()
            self.ER = self.calcEx(np.reshape(q_exa_bdy[-3:],(3,1))).flatten()



    ######################################
    ''' Begin  defining class functions'''
    ######################################


    def fun_s(self, xvec):
        ''' Defines the shape of the nozzle. '''
        
        # Nozzle from Hirsch and Zingg
        def fun_s_book(xvec):
            def fun_s_part1(x):
                    return 1 + 1.5*(1-x/5)**2    
            def fun_s_part2(x):
                return 1 + 0.5*(1-x/5)**2    
            svec = np.where(xvec<5,fun_s_part1(xvec),fun_s_part2(xvec))
            return svec
        
        # Nozzle with constantly changing shape (constant dsdx)
        def fun_s_linear(xvec):
            svec = 2 - xvec/10
            return svec
        
        # Nozzle of constant shape (dsdx=0)
        def fun_s_const(xvec):
            svec = np.ones(xvec.shape)
            return svec
        
        # Smooth Nozzle approximating book shape
        def fun_s_smooth(xvec):
            svec = -1/250*xvec**3 + 1/10*xvec**2 - 7/10*xvec + 5/2
            return svec

        if self.nozzle_shape == 'book':
            return fun_s_book(xvec)
        elif self.nozzle_shape == 'constant':
            return fun_s_const(xvec)
        elif self.nozzle_shape == 'linear':
            return fun_s_linear(xvec)
        elif self.nozzle_shape == 'smooth':
            return fun_s_smooth(xvec)
        else:
            raise Exception('Unknown nozzle shape')

    def fun_der_s(self, xvec):
        ''' Defines the derivative of the nozzle. '''
        
        # Nozzle from Hirsch and Zingg
        def fun_der_s_book(xvec):
            def fun_der_s_part1(x):
                return -(3/5)*(1-x/5)  
            def fun_der_s_part2(x):
                return -(1/5)*(1-x/5)   
            svec = np.where(xvec<5,fun_der_s_part1(xvec),fun_der_s_part2(xvec))
            return svec
        
        # Nozzle with constantly changing shape (constant dsdx)
        def fun_der_s_linear(xvec):
            svec = -np.ones(xvec.shape)/10
            return svec
        
        # Nozzle of constant shape (dsdx=0)
        def fun_der_s_const(xvec):
            svec = -np.zeros(xvec.shape)
            return svec
    
        # Smooth Nozzle approximating book shape
        def fun_der_s_smooth(xvec):
            svec = -1/250*3*xvec**2 + 1/10*2*xvec - 7/10
            return svec
    
        if self.nozzle_shape == 'book':
            return fun_der_s_book(xvec)
        elif self.nozzle_shape == 'constant':
            return fun_der_s_const(xvec)
        elif self.nozzle_shape == 'linear':
            return fun_der_s_linear(xvec)
        elif self.nozzle_shape == 'smooth':
            return fun_der_s_smooth(xvec)
        else:
            raise Exception('Unknown nozzle shape')

      
    def calc_p(self,rho,rhou,e):
        ''' function to calculate the pressure given Q variables '''
        # Note: If fed in variables Q*S instead of Q, will return p*S instead of p, as desired
        return (self.g-1)*(e-0.5*(rhou*rhou/rho))

    def calc_a(self,rho,rhou,e):
        ''' function to calculate the sound speed given pressure and Q1 '''
        # Note: Regardless if fed Q*S or Q, will always return a, not a*S
        return np.sqrt(self.g*self.calc_p(rho,rhou,e)/rho)    

    def decompose_q(self, q):
        ''' splits q[nen*neq_node,nelem] or q[nen*neq_node] to q_i[nen,nelem] 
        or q_i[nen] for each i in neq_node '''

        q_0 = q[0::self.neq_node]
        q_1 = q[1::self.neq_node]
        q_2 = q[2::self.neq_node]

        return q_0, q_1, q_2

    @staticmethod
    def assemble_vec(input_vec):
        ''' assembles q_i[nen,nelem] or q_i[nen] to q[nen*neq_node,nelem] 
        or q[nen*neq_node] for each i in neq_node '''

        nvec = len(input_vec)   # No. of ind. vectors
        vec_shape = np.array(input_vec[0].shape) # (nen,nelem) or (nen,)
        vec_shape[0] *= nvec   # (nen*neq_node,nelem) or (nen*neq_node,)
        vec = np.stack(input_vec).reshape(vec_shape, order='F')

        return vec

    @staticmethod
    def prim2cons(rho, u, e, svec):
        '''takes y_i[nen,nelem] or y_i[nen] primitive variables (for each i) to 
        q[nen*neq_node,nelem] or q[nen*neq_node] conservative vector, as long 
        as svec is in a shape that matches y_i '''
        
        q_0 = rho * svec
        q_1 = q_0 * u
        q_2 = e * svec

        q = Quasi1dEuler.assemble_vec((q_0, q_1, q_2))

        return q

    def cons2prim(self, q, svec):
        '''takes q[nen*neq_node,nelem] or q[nen*neq_node] conservative vector 
        to y_i[nen,nelem] or y_i[nen] primitive variables (for each i), as long 
        as svec is in a shape that matches y_i '''

        q_0, q_1, q_2 = self.decompose_q(q)

        rho = q_0 / svec
        u = q_1 / q_0
        e = q_2 / svec
        P = (self.g-1)*(e - (rho * u**2)/2)

        a = np.sqrt(self.g * P/rho)

        # Convert numpy array with 1 node to scalar values
        if q.size == 3 and q.ndim==1:
            rho = rho[0]
            u = u[0]
            e = e[0]
            P = P[0]
            a = a[0]
        return rho, u, e, P, a
    
    def entropy(self, q):
        ''' return the nodal values of the entropy s(q). '''
        # Note: this is not quite the "normal" entropy for quasi1D euler when \neq 1, but is a correct entropy
        q_0, q_1, q_2 = self.decompose_q(q)
        u = q_1 / q_0
        p = (self.g-1)*(q_2 - q_0*0.5*u*u) # p*S
        s = np.log(p/(q_0**self.g))
        return (-q_0*s/(self.g-1))
    
    def entropy_var(self, q):
        ''' return the nodal values of the entropy variables w(q). '''
        # Note: the same entropy variables for quasi1D euler (no svec dependence)
        q_0, q_1, q_2 = self.decompose_q(q)
        u = q_1 / q_0
        e = q_2 
        p = (self.g-1)*(e - q_0*0.5*u*u) # = p*S
        s = np.log(p/(q_0**self.g))
        w = self.assemble_vec(((self.g-s)/(self.g-1) - 0.5*q_0*u**2/p, q_0*u/p, -q_0/p))
        return w

    def exact_sol(self, time=0, x=None, extra_vars=False, nondimensionalize=None):
        ''' Returns the exact solution at given time. Use default time=0 for
        steady solutions. if extra_vars=True, a dictionary with arrays for
        mach, T, p, rho, a, u, and e is also returned along with exa_sol. 
        NOTE: some of these functions are not properly vectorized... I don't care.
              I wrote them a long time ago and am too lazy to fix now. '''
        ntol=1e-14 # tolerance for numerical solution of equations

        if nondimensionalize is None:
            nondimensionalize = self.nondimensionalize

        if self.nondimensionalize:
            time /= self.t_scale
        
        if x is None:
            x = self.x_elem   
        svec = self.fun_s(x)
            
        def calc_T(T_in, mach_in):
            ''' calculate temperature from isentropic relations. Zingg 3.46 '''
            k = 1 + (self.g-1)/2 * mach_in**2
            T = T_in / k
            return T

        def calc_p(p_in, mach_in):
            ''' calculate pressure from isentropic relations. Zingg 3.47 '''
            k = 1 + (self.g-1)/2 * mach_in**2
            p = p_in * k**(-self.g/(self.g-1))
            return p

        def RankineHugoniot(machL, s_critL, T0L, p0L):
            ''' Calcualte values across a shock. Given left (upstream) values,
            returns right (downstream) values. See Zingg 3.48-3.52 '''    
            # useful constants
            k1 = (self.g+1)*machL**2
            k2 = (self.g-1)*machL**2
            #k3 = 2 * self.g * machL**2 - (self.g - 1)
            k4 = ((k1/2)/(1+(k2/2)))**(self.g/(self.g-1))
            k5 = ((2*self.g/(self.g+1)) * machL**2 - (self.g-1)/(self.g+1))**(1/(self.g-1))
            k6 = (self.g+1)/(2*(self.g-1))
            
            # temperature (3.48) where T0L=self.T01
            T0R = T0L

            # mach speed (3.49)
            # machR = (2 + k2)/k3
            
            # static pressure (3.50)
            # pR = pL * (k3/(self.g + 1))
            
            # total pressure (3.51) where p0L=self.p01
            p0R = p0L * (k4/k5)
            
            # critical area (3.52) where T0L=self.T01, p0L=self.p01
            rho01 = p0L / (self.R * T0L)
            rho0R = p0R / (self.R * T0L)
            a01 = np.sqrt(self.g * p0L / rho01)
            a0R = np.sqrt(self.g * p0R / rho0R)
            rho_a_L_star = rho01*a01*(2/(self.g+1))**k6
            rho_a_R_star = rho0R*a0R*(2/(self.g+1))**k6
            s_critR = s_critL * rho_a_L_star / rho_a_R_star

            return s_critR, T0R, p0R
        
        def solve_mach(mach_in, svec_in, s_crit):
            '''given an initial guess for the mach number, iteratively solve for
            the correct mach number for a given svec and s_crit. See Zingg 3.45
            Used for subsonic and transonic exact solutions.'''
            
            guess = mach_in*np.ones(svec_in.shape)

            def fun_mach(mach, s_vec=svec_in):
                # Zingg 3.45
                k = (2/(self.g+1)) * (1 + 0.5*(self.g-1) * mach**2)
                exp = (self.g+1)/(2*(self.g-1))
                return k**exp - s_vec * mach / s_crit
    
            def der_fun_mach(mach, s_vec=svec_in):
                # derivative of Zingg 3.45 w.r.t. mach
                k = (2/(self.g+1)) * (1 + 0.5*(self.g-1) * mach**2)
                exp = (self.g+1)/(2*(self.g-1))
                return mach*k**(exp-1) - s_vec / s_crit
            
            results = newton(fun_mach, guess, fprime=der_fun_mach, 
                             tol=ntol,maxiter=1000,full_output=True,disp=False)
            
            if len(results) == 2: # scalar case. see scipy.optimize.newton
                mach_calc, converged = results[0], results[1].converged
                if not converged:
                    print('exact_sol: Not all values converged. Trying again with bisect.')
                    if mach_in < 1: lim = [0,1]
                    else: lim = [1,2]
                    mach_calc, res = bisect(fun_mach,lim[0],lim[1],args=(svec_in),xtol=ntol,
                                       maxiter=1000,full_output=True,disp=False)
                    if not res.converged:
                        print('WARNING exact_sol: failed solve_mach for svec =',svec_in)
            elif len(results) == 3: # ndarray case. see scipy.optimize.newton
                mach_calc, converged = results[0], results[1]
                if np.any(np.invert(converged)):
                    print('exact_sol: Not all values converged. Trying again with bisect.')
                    idx = np.linspace(0,len(svec_in)-1,len(svec_in))[np.invert(converged)]
                    if mach_in < 1: lim = [0,1]
                    else: lim = [1,2]
                    for i in idx:
                        mach_calc[int(i)], res = bisect(fun_mach,lim[0],lim[1],
                                args=(svec_in[int(i)]),xtol=ntol,maxiter=1000,
                                full_output=True,disp=False)
                        if not res.converged:
                            print('WARNING exact_sol: failed solve_mach for svec =',svec_in[int(i)])
            
            return mach_calc
    
        def solve_pressure(aL, aR, pL, pR):
            '''given an initial guess for the pressure ratio, iteratively solve
            for the correct pressure accross a shock for given air speeds on 
            either side of the shock. See Zingg 3.54
            Used for shock tube exact solutions.'''

            def fun_press(P):
                # Zingg 3.54
                alpha = (self.g + 1) / (self.g - 1)
                lhs = np.sqrt(2/(self.g *(self.g-1))) * (P-1)/np.sqrt(1+alpha*P)
                rhs = (2/(self.g-1)) * (aL/aR) * (1 - (P*pR/pL)**((self.g-1)/(2*self.g)))
                return lhs - rhs

            press_calc, res = bisect(fun_press,1,10,xtol=ntol,maxiter=1000,
                                       full_output=True,disp=False)
            if not res.converged:
                print('WARNING exact_sol: failed solve_pressure')
            return press_calc
        
        def remaining_param(p, T, mach):
            ''' calculate density, sound speed, velocity, and energy once
            total pressure, temperature, and mach number are known '''
            rho = p / (self.R * T)
            a = np.sqrt(self.g * self.R * T)
            u = a * mach
            e = p/(self.g-1) + rho * u**2 /2
            return rho, a, u, e
        
        def channel_flow(T01, p01, svec, s_crit, mach_guess):
            ''' function to solve a channel flow exactly (sub/supersonic).
            Requires:   svec, s_crit = nozzle shape and critical area
                        self.R , self.g = flow variables
                        T01 , p01 = total temp and pressure at the inlet
                        mach_guess = >1 for supersonic or <1 for subsonic'''
            # since there is no shock, we only need to compute mach once
            mach = solve_mach(mach_guess, svec, s_crit)
            T = calc_T(T01, mach)
            p = calc_p(p01, mach)
            return mach, T, p

        def transonic(svec):
            ''' function to solve the transonic flow exactly.
            Requires:   xx = flattened array of node locations in x
                        self.shockx = x location of shock
                        svec, self.s_crit = nozzle shape and critical area
                        self.R , self.g = flow variables
                        T01 , p01 = total temp and pressure at the inlet '''
            # get index for shock location, shockx. If is not exact, add it and delete later.
            shocki = np.nonzero((abs(x-self.shockx))<1e-10)[0]
            # get index for transition to supersonic flow (mach number = 1, or svec=s_crit).
            transi = np.nonzero((abs(x-self.transx))<1e-10)[0]
            
            ''' check returned indices. If no indices were given, svec must be
                modidified by adding a node at the location, then later deleted.
                If a unique node exists there, treat it as if it were on the LHS.
                If 2 nodes exist there, treat one on either side of the location.
                If >2 nodes esist there, raise an Exception. '''
            del_shocki = False
            if shocki.size == 0:
                shocki = np.where(x>self.shockx)[0][0] + 1 # +1 because of slicing later on 
                shocks = self.fun_s(np.array([self.shockx]))[0] # get svec value at shocki
                svec = np.insert(svec,shocki-1,shocks) # insert shocks at shocki
                del_shocki = True
            elif shocki.size == 1: shocki = shocki[0] + 1   # +1 because of slicing later on 
            elif shocki.size == 2:
                assert(shocki[0]+1==shocki[1]),'Nodes at shock are not adjacent.'
                shocki = shocki[0] + 1 
            else: raise Exception('The grid has >2 nodes at shock.')
            
            del_transi = False
            if transi.size == 0:                      
                transi = np.where(x>self.transx)[0][0] + 1 # +1 because of slicing later on 
                transs = self.fun_s(np.array([self.transx]))[0] # get svec value at transi
                svec = np.insert(svec,transi-1,transs) # insert transs at transi
                del_transi = True
                shocki += 1 # if we add a transition node, the shock node gets shifted over
            elif transi.size == 1: transi = transi[0] + 1   # +1 because of slicing later on 
            elif transi.size == 2:
                assert(transi[0]+1==transi[1]),'Nodes at transition are not adjacent.'
                transi = transi[0] + 1 
            else: raise Exception('The grid has >2 nodes at transition.')
            
            # initialize arrays for Mach number, temperature, and pressure
            mach = np.empty(len(svec))
            T = np.empty(len(svec))
            p = np.empty(len(svec))
            
            # do for subsonic and supersonic part before shock
            mach[:transi], T[:transi], p[:transi] = channel_flow(self.T01, self.p01,
                        svec[:transi], self.s_crit, 0.3)
            mach[transi:shocki], T[transi:shocki], p[transi:shocki] = channel_flow(self.T01, self.p01,
                        svec[transi:shocki], self.s_crit, 1.2)
    
            # for part after shock, use the Rankine-Hugoniot relations
            s_critR, T0R, p0R = RankineHugoniot(mach[shocki-1], self.s_crit, self.T01, self.p01)
            mach[shocki:], T[shocki:], p[shocki:] = channel_flow(T0R, p0R, svec[shocki:], s_critR, 0.3)
            
            # if we added any nodes, now delete them
            if del_transi:
                mach = np.delete(mach,transi)
                T = np.delete(T,transi)
                p = np.delete(p,transi)
                svec = np.delete(svec,transi)
                shocki -= 1 # if we delete a transition node, the shock node gets shifted over
            if del_shocki:
                mach = np.delete(mach,shocki)
                T = np.delete(T,shocki)
                p = np.delete(p,shocki)
                svec = np.delete(svec,shocki)
            
            return mach, T, p
        
        def shocktube(tf):
            ''' function to solve the sod shock tube Riemann problem exactly.
            Requires:   xx = flattened array of node locations in x
                        self.xmembrane = x location of membrane
                        self.pL, self.rhoL, self.pR, self.rhoR = left and right initial states
                        tf = time to calculate solution at
                        self.R , self.gamma = flow variables '''
            
            assert(np.max(abs(svec-1))<1e-10),'svec must be =1 for shock tube'

            # Set remaining initial variables
            aL = np.sqrt(self.g*self.pL/self.rhoL)
            aR = np.sqrt(self.g*self.pR/self.rhoR)

            # get pressure and density in region 2 (between contact surface and shock)
            Pratio = solve_pressure(aL, aR, self.pL, self.pR)
            p2 = Pratio*self.pR # stated before Zingg 3.54
            rho2 = self.rhoR*(1+(self.g+1)*Pratio/(self.g-1))/((self.g+1)/(self.g-1)+Pratio) # Zingg 3.55

            # get pressure and density in region 3 (between expansion fan and contact surface)
            p3 = p2 # stated before Zingg 3.56
            rho3 = self.rhoL*(p3/self.pL)**(1/self.g) # Zingg 3.57

            # get propagation speed of the contact surface
            V = 2*aL*(1-(p3/self.pL)**((self.g-1)/2/self.g))/(self.g-1) # Zingg 3.56

            # get shock propagation speed
            C = (Pratio-1)*aR**2/self.g/V # Zingg 3.58

            # set boundary values of x for regions L,5,3,2,R
            xL5 = self.xmembrane - aL*tf
            x53 = self.xmembrane + (V*(self.g+1)/2 -aL)*tf
            x32 = self.xmembrane + V*tf
            x2R = self.xmembrane + C*tf

            # initialize arrays
            mach = np.empty(len(x))
            p = np.empty(len(x))
            rho = np.empty(len(x))
            a = np.empty(len(x))
            u = np.empty(len(x))
        
            for i in range(len(x)):
                if x[i] < xL5 + 1e-12: # region L
                    rho[i] = self.rhoL
                    mach[i] = 0
                    p[i] = self.pL
                    #a[i] = np.sqrt(self.g*self.pL/self.rhoL)
                    #u[i] = 0
                elif x[i] < x53 + 1e-12: # get pressure and density in region 5 (expansion fan)
                    u[i] = 2*((x[i]-self.xmembrane)/tf+aL)/(self.g+1)
                    a[i] = u[i] - (x[i]-self.xmembrane)/tf
                    p[i] = self.pL*(a[i]/aL)**(2*self.g/(self.g-1))
                    rho[i] = self.g*p[i]/a[i]**2
                    mach[i] = u[i]/a[i]
                elif x[i] < x32 + 1e-12: # region 3 (between expansion fan and contact surface)
                    rho[i] = rho3
                    p[i] = p3
                    a[i] = np.sqrt(self.g*p3/rho3)
                    mach[i] = V/a[i]
                    #u[i] = V
                elif x[i] < x2R + 1e-12: # region 2 (between contact surface and shock)
                    rho[i] = rho2
                    p[i] = p2
                    a[i] = np.sqrt(self.g*p2/rho2)
                    mach[i] = V/a[i]
                    #u[i] = V
                elif x[i] < x[-1] + 1e-12: # region R
                    rho[i] = self.rhoR
                    p[i] = self.pR
                    mach[i] = 0
                    #a[i] = np.sqrt(self.g*self.pR/self.rhoR)
                    #u[i] = 0
                else:
                    print('x indexing out of range. You messed up buddy')
    
            # get temperature distribution
            T = p/self.R/rho
            return mach, T, p
        
        def density_wave(tf):
            ''' function to solve a simple travelling wave.
            Requires:   xx = flattened array of node locations in x
                        self.xmin, self.xmax, self.dom_len = important x values
                        self.u0, self.p0 = constant velocity and pressure
                        tf = time to calculate solution at
                        self.R , self.gamma = flow variables '''
            
            assert(np.max(abs(svec-1))<1e-10),'svec must be =1 for wave solution'
            if self.q0_type != 'density_wave':
                print("ERROR: for exact_sol, initial condition must be density_wave, not '"+self.q0_type+"'")
                return np.zeros(np.shape(x)), np.zeros(np.shape(x)), np.zeros(np.shape(x))
            
            # just like linear convection equation. See Gassner et al 2020
            xy_mod = np.mod((x - self.xmin) - self.u0*tf, self.dom_len) + self.xmin
            q = self.set_q0(xy=xy_mod)
            rho, u, e, p, a = self.cons2prim(q, svec)
            T = p / (rho * self.R)
            mach = u / a
            return mach, T, p
        
        def manufactured_soln(t):
            ''' manufactured solution. '''
            
            assert(np.max(abs(svec-1))<1e-10),'svec must be =1 for wave solution'
            if self.q0_type != 'manufactured_soln':
                print("ERROR: for exact_sol, initial condition must be manufactured_soln, not '"+self.q0_type+"'")
                return np.zeros(np.shape(x)), np.zeros(np.shape(x)), np.zeros(np.shape(x))
            
            rho = 2. + 0.1*np.sin(np.pi*(x-t))
            u = np.ones_like(x)
            e = rho*rho
            p = (self.g - 1)*(e - 0.5*rho)
            a =  np.sqrt(self.g*p/rho)
            T = p / (rho * self.R)
            mach = u / a
            return mach, T, p

        reshape=False
        if self.test_case == 'subsonic_nozzle':
            if x.ndim >1: 
                reshape=True
                x = x.flatten('F')
                svec = svec.flatten('F')
            mach, T, p = channel_flow(self.T01, self.p01, svec, self.s_crit, 0.3)           
        elif self.test_case == 'transonic_nozzle':
            if x.ndim >1: 
                reshape=True
                x = x.flatten('F')
                svec = svec.flatten('F')
            mach, T, p = transonic(svec)           
        elif self.test_case == 'shock_tube':
            if x.ndim >1: 
                reshape=True
                x = x.flatten('F')
                svec = svec.flatten('F')
            mach, T, p = shocktube(time)
        elif self.test_case == 'density_wave':
            mach, T, p = density_wave(time)
        elif self.test_case == 'manufactured_soln':
            mach, T, p = manufactured_soln(time)
        else:
            raise Exception('Invalid test case.')
            
        if nondimensionalize:
            p = p / (self.rho_inf*self.a_inf*self.a_inf)
            T = T / (self.a_inf*self.a_inf)
        rho, a, u, e = remaining_param(p,T,mach)
        exa_sol = self.prim2cons(rho, u, e, svec)

        if reshape: 
            #svec = np.reshape(svec, (self.nen,self.nelem), 'F') # fails if x was not qshape
            exa_sol = np.reshape(exa_sol,(self.nen*self.neq_node,self.nelem),'F')
        
        if extra_vars:
            var_names = ['mach', 'T', 'p', 'rho', 'a', 'u', 'e']
            exa_sol_extra = {}
            for i in range(len(var_names)):
                values = eval(var_names[i])
                if reshape: values = np.reshape(values,(self.nen,self.nelem),'F')
                exa_sol_extra[var_names[i]] = values 
            return exa_sol, exa_sol_extra
        else: return exa_sol

    def set_mesh(self, mesh):
        ''' Overwrite base function in DiffEqBase '''

        PdeBase.set_mesh(self, mesh)

        # Calculate the shape of the nozzle at the mesh and boundary nodes
        self.svec = self.fun_s(self.x)
        self.svec_der = self.fun_der_s(self.x)
        self.svec_elem = np.reshape(self.svec,(self.nen,self.nelem),'F')
        self.svec_der_elem = np.reshape(self.svec_der,(self.nen,self.nelem),'F')
            

    def set_q0(self, q0_type=None, xy=None):
        # overwrite base function from PdeBase
        
        if q0_type is None:
            q0_type = self.q0_type
        q0_type = q0_type.lower()

        if xy is None:
            xy = self.x_elem
            svec = self.svec_elem
        else:
            svec = self.fun_s(xy)

        if self.test_case == 'density_wave':
            if q0_type != 'density_wave':
                print("WARNING: Instead of using q0_type = '"+q0_type+", you should probably use q0_type = 'density_wave'.")
                q0 = PdeBase.set_q0(self, q0_type=q0_type, xy=xy)
                fn.repeat_neq_gv(q0,self.neq_node)
            else:
                rho = 1 + 0.98*np.sin(2*np.pi*xy)
                u = self.u0 * np.ones(rho.shape)
                p = self.p0 * np.ones(rho.shape)
                e = p/(self.g-1) + rho * u**2 /2
                q0 = self.prim2cons(rho, u, e, svec)

        elif self.test_case == 'manufactured_soln':
            assert(q0_type == 'manufactured_soln'),"Must use q0_type == 'manufactured_soln'."
            rho = 2 + 0.1*np.sin(np.pi*xy)
            u = np.ones(rho.shape)
            e = rho*rho
            p = (self.g-1)*rho*(rho-0.5)
            q0 = self.prim2cons(rho, u, e, svec)
                
        elif self.test_case == 'shock_tube':
            if q0_type != 'shock_tube':
                print('WARNING: Ignoring q0_type and setting initial conditions to shock_tube.')
            q0 = self.exact_sol(time=0, x=xy, nondimensionalize=False)
            
        else: # subsonic or transonic
            if q0_type == 'exact':
                q0 = self.exact_sol(time=0, x=xy, nondimensionalize=False)
    
            elif q0_type == 'linear':
                if self.nondimensionalize: # must redimensionalize first
                    rhoL, rhoR = self.rhoL*self.rho_inf, self.rhoR*self.rho_inf
                    uL, uR = self.uL*self.a_inf, self.uR*self.a_inf
                    eL, eR = self.eL*self.rho_inf*self.a_inf**2, self.eR*self.rho_inf*self.a_inf**2
                else:
                    rhoL, rhoR = self.rhoL, self.rhoR
                    uL, uR = self.uL, self.uR
                    eL, eR = self.eL, self.eR
                rho = (xy - self.xmin)/(self.xmax - self.xmin)*rhoR + (xy - self.xmax)/(self.xmin - self.xmax)*rhoL
                u = (xy - self.xmin)/(self.xmax - self.xmin)*uR + (xy - self.xmax)/(self.xmin - self.xmax)*uL
                e = (xy - self.xmin)/(self.xmax - self.xmin)*eR + (xy - self.xmax)/(self.xmin - self.xmax)*eL
                q0 = self.prim2cons(rho, u, e, svec)

            else:
                print("WARNING: Instead of using q0_type = '"+q0_type+"', you should probably use q0_type = 'linear' or 'exact'.")
                q0 = PdeBase.set_q0(self, q0_type=q0_type, xy=xy)
                fn.repeat_neq_gv(q0,self.neq_node)
        
        if self.nondimensionalize:
            q0[0::3] = q0[0::3] / self.rho_inf
            q0[1::3] = q0[1::3] / (self.rho_inf*self.a_inf)
            q0[2::3] = q0[2::3] / (self.rho_inf*self.a_inf*self.a_inf)

        return q0


    def var2plot(self, q_in, var2plot_name):

        q = np.copy(q_in)
        if self.nondimensionalize:
            # re-dimensionalize
            q[0::3] = q[0::3] * self.rho_inf
            q[1::3] = q[1::3] * (self.rho_inf*self.a_inf)
            q[2::3] = q[2::3] * (self.rho_inf*self.a_inf*self.a_inf)

        svecfail = False
        if q.ndim == 1:
            svec = self.svec
            if len(q) != self.nn: svecfail = True
        else:
            svec = self.svec_elem
            if q.shape[1] != self.nelem: svecfail = True
        if svecfail:
            print('WARNING: q given to var2plot is not the same shape as self.svec_elem')
            print('         will default to plotting q[0], which may be rho or rho*S')
            print('         ', q.shape, self.svec.shape, self.svec_elem.shape)
            return q[::self.neq_node]
        
        if var2plot_name is None:
            var2plot_name = self.plt_var2plot_name
        rho, u, e, P, a = self.cons2prim(q, svec)
        w = self.entropy_var(q)

        if var2plot_name == 'rho' or var2plot_name == r'$\rho$':
            return rho
        elif var2plot_name == 'u' or var2plot_name == r'$u$':
            return u
        elif var2plot_name == 'rhou' or var2plot_name == r'$\rho u$':
            return rho*u
        elif var2plot_name == 'e' or var2plot_name == r'$e$':
            return e
        elif var2plot_name == 'p' or var2plot_name == r'$p$':
            return P
        elif var2plot_name == 'a' or var2plot_name == r'$a$':
            return a
        elif var2plot_name == 'mach' or var2plot_name == r'$M$' or var2plot_name == 'Ma':
            return u / a
        elif var2plot_name == 'w1' or var2plot_name == r'$w_1$':
            return w[::3]
        elif var2plot_name == 'w2' or var2plot_name == r'$w_2$':
            return w[1::3]
        elif var2plot_name == 'w3' or var2plot_name == r'$w_3$':
            return w[2::3]
        elif var2plot_name == 's' or var2plot_name == r'$s$' or var2plot_name == 'entropy':
            return self.entropy(q)
        else:
            raise Exception('Requested variable to plot is not available, '+var2plot_name)

    def calcEx(self, q):

        q_0, q_1, q_2 = self.decompose_q(q)
        u = q_1 / q_0

        k = u*q_1                    # Common term: rho * u^2 * S
        ps = (self.g-1)*(q_2 - k/2)  # p * S

        e0 = q_1
        e1 = k + ps
        e2 = u*(q_2 + ps)

        E = self.assemble_vec((e0, e1, e2))
        return E
    
    def dExdq(self, q):

        q_0, q_1, q_2 = self.decompose_q(q)
        u = q_1 / q_0        
        u2 = u**2
        q2_q0 = q_2/q_0

        # entries of the dEdq (A) matrix
        if np.any(np.iscomplex(q)):
            r1 = np.ones(q_0.shape, dtype=complex)
            r0 = np.zeros(q_0.shape, dtype=complex)
        else:
            r1 = np.ones(q_0.shape)
            r0 = np.zeros(q_0.shape)
        r21 = 0.5*(self.g-3) * u2
        r22 = (3-self.g) * u
        r23 = r1*(self.g-1)
        r31 = (self.g-1) * (u * u2) - self.g * q2_q0 * u
        r32 = self.g * q2_q0 - (3/2)*(self.g-1) * u2
        r33 = self.g * u
        
        dEdq = fn.block_diag(r0,r1,r0,r21,r22,r23,r31,r32,r33)

        return dEdq
        
    def dqdw(self,q):
        ''' return hessian P of potential phi wrt entropy variables w, or dqdw '''
        # returns S * P if quasi1D Euler
        rho, rhou, e = self.decompose_q(q) # = (rho, rho*u, e) * S if quasi1D Euler
        rhou2 = rhou*rhou/rho # = rho * u^2 * S if quasi1D Euler
        p = (self.g-1)*(e-rhou2/2) # pressure = p * S if quasi1D Euler
        
        r22 = rhou2 + p
        r23 = rhou*(p+e)/rho
        r33 = self.g*e*e/rho - ((self.g-1)/4)*rhou2*rhou2/rho
        
        P = fn.block_diag(rho,rhou,e,rhou,r22,r23,e,r23,r33)
        return P
    
    def calcG(self, q, t):
        q_0, q_1, q_2 = self.decompose_q(q)
        p = (self.g-1)*(q_2 - 0.5*q_1*q_1 /q_0) / self.svec_elem
        g2 = p * self.svec_der_elem
        zero_vec = np.zeros((self.nen,self.nelem))
        G = self.assemble_vec((zero_vec, g2, zero_vec))
        return G
    
    def calcG_manufactured(self, q, t):
        tmod = t/self.t_scale
        h = (2. + 0.1*np.sin(np.pi*(self.x_elem-tmod)))
        dhdt = -0.1*np.pi*np.cos(np.pi*(self.x_elem-tmod))
        dpdx = -dhdt*(2.*h-0.5)*(self.g-1)/(self.rho_inf*self.a_inf*self.a_inf)
        # for this problem, rho = h, dhdt = -dhdx, u=1
        # if you work it out, the convective derivatives always cancel
        G = np.zeros_like(q)
        G[1::3,:] = dpdx
        G[2::3,:] = dpdx/self.a_inf
        return G
    
    def dGdq(self, q):
        q_0, q_1, _ = self.decompose_q(q)
        fac = q_1 / q_0
        r23 = (self.g - 1) * self.svec_der_elem / self.svec_elem
        r22 = -r23 * fac
        r21 = r23 * fac * fac / 2

        nen = self.nen * 3
        # TODO: Actually faster if we leave it dense... 
        #dGdq = sp.csr_matrix((self.nn * 3, self.nn * 3))
        dGdq = np.zeros((self.nn * 3, self.nn * 3))

        for e in range(self.nelem):
            start = nen * e
            for n in range(self.nen):
                row = start + n * 3 + 1
                col0 = start + n * 3
                col1 = start + n * 3 + 1
                col2 = start + n * 3 + 2
                
                dGdq[row, col0] = r21[n, e]
                dGdq[row, col1] = r22[n, e]
                dGdq[row, col2] = r23[n, e]

        return dGdq

    def d2Exdq2(self, q):
        return None

    def maxeig_dExdq(self, q):
        ''' return the maximum eigenvalue - used for LF fluxes. accepts q[:,:] and q[:] '''
        rhoS = q[::3] 
        u = q[1::3]/rhoS 
        e_rho = q[2::3]/rhoS 
        p_rho = (self.g-1)*(e_rho-0.5*u*u) # pressure / rho, even if quasi1D Euler
        a = np.sqrt(self.g*p_rho) # sound speed, = a if quasi1D Euler
        lam = np.maximum(np.abs(u+a),np.abs(u-a))
        return lam
    
    def dissipation_something(self,q):
        ''' some entropy-stable volume dissipation '''
        raise Exception('Not coded up yet')
        
    def dDdq_something(q):
        ''' linearization of volume dissipation '''
        raise Exception('Not coded up yet')

    '''
    def dfdq(self, q,  xy_idx0=None, xy_idx1=None):
        A = self.dEdq(q)
        dGdq = self.dGdq(q, xy_idx0, xy_idx1)

        dfdq = -self.der1 @ A + dGdq
        return dfdq    
    '''
    
    def Ismail_Roe(self, qL,qR):
        '''
        Return the ismail roe flux given two states uL and rR where each is
        of shape (neq=3,), and returns a numerical flux of shape (neq=3,)
        note subroutine defined in ismail-roe appendix B for logarithmic mean
        '''
        g = self.g
        
        rhoL, rhoR = qL[0], qR[0]
        uL, uR = qL[1]/rhoL, qR[1]/rhoR
        pL, pR = (g-1)*(qL[2] - (rhoL * uL**2)/2), (g-1)*(qR[2] - (rhoR * uR**2)/2)

        alphaL = np.sqrt(rhoL/pL)
        alphaR = np.sqrt(rhoR/pR)
        betaL = np.sqrt(rhoL*pL)
        betaR = np.sqrt(rhoR*pR)

        xi_alpha = alphaL/alphaR
        zeta_alpha = (1-xi_alpha)/(1+xi_alpha)
        zeta_alpha2 = zeta_alpha**2
        if zeta_alpha2 < 0.01:
            F_alpha = 2*(1. + zeta_alpha2/3. + zeta_alpha2**2/5. + zeta_alpha2**3/7.)
        else:
            F_alpha = - np.log(xi_alpha)/zeta_alpha
        alpha_ln = (alphaL+alphaR)/F_alpha

        xi_beta = betaL/betaR
        zeta_beta = (1-xi_beta)/(1+xi_beta)
        zeta_beta2 = zeta_beta**2
        if zeta_beta2 < 0.01:
            F_beta = 2*(1. + zeta_beta2/3. + zeta_beta2**2/5. + zeta_beta2**3/7.)
        else:
            F_beta = - np.log(xi_beta) / zeta_beta
        beta_ln = (betaL+betaR)/F_beta

        alpha_avg = 0.5*(alphaL+alphaR)
        beta_avg = 0.5*(betaL+betaR)

        rho_avg = alpha_avg * beta_ln
        a_avg2 = (0.5/rho_avg)*((g+1)*beta_ln/alpha_ln + (g-1)*beta_avg/alpha_avg)
        u_avg = 0.5 * (uL*alphaL + uR*alphaR) / alpha_avg
        p_avg = beta_avg / alpha_avg
        H_avg = a_avg2/(g - 1) + 0.5*u_avg**2
        
        rhou_avg = rho_avg*u_avg
        return np.array([rhou_avg, rhou_avg*u_avg + p_avg, rhou_avg*H_avg]) 