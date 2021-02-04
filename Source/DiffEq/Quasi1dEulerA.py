#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 23:51:27 2021

@author: bercik
"""

import numpy as np
#import scipy.sparse as sp
#from scipy.sparse.linalg import spsolve
from scipy.optimize import newton, bisect

from Source.DiffEq.DiffEqBase import PdeBaseCons
from Source.DiffEq.SatBase import SatBaseCons
import Source.Methods.Functions as fn

class Quasi1dEuler(PdeBaseCons):
    
    # Equation constants
    diffeq_name = 'Quasi1dEuler'
    dim = 1
    neq_node = 3            # No. of equations per node
    npar = 0                # No. of design parameters
    has_exa_sol = True

    # Problem constants
    R = 287
    g = 1.4 # gamma

    # Plotting constants
    plt_var2plot_name = 'rho' # rho, u, e, p, a, mach    

    # TODO: Initial and boundary conditions
    bc_type = 'dirichlet'     # 'dirichlet', 'riemann'

    # TODO: Parameters for the solvers
    use_local_dt = False

    # Normalizing the solution
    norm_loc = 'left'

    # Plotting
    calc_exa_sol = True

    # TODO
    #self.svec= flattened like xx.
    #self.svec_elem = like q_sol # maybe not needed

    def __init__(self, para=None, obj_name=None, q0_type=None, test_case='subsonic',
                 nozzle_shape='book', norm_var=True):

        ''' Add inputs to the class '''

        super().__init__(para, obj_name, q0_type)
        self.test_case = test_case
        self.nozzle_shape = nozzle_shape
        self.norm_var = norm_var
        
        if self.q0_type == None:
            self.q0_type = 'linear' # can be exact, linear

        ''' Set flow and problem dependent parameters  '''
        
        if self.test_case == 'subsonic':
            self.xmin_nozzle = 0  # may differ from self.x_min if normalizing
            self.xmax_nozzle = 10 # may differ from self.x_max if normalizing
            self.T01 = 300         # Temperature in Kelvin at inlet
            self.p01 = 100*1000   # Total Pressure in Pa at inlet
            self.s_crit = 0.8     # critical nozzle area. mach=1 when svec=s_crit
            self.k2 = 0.          # coefficient for first order FD dissipation, usually 0
            self.k4 = 0.02        # coefficient for third order FD dissipation, usually 0.02
            self.isperiodic = False
            self.steady = True
            
        elif self.test_case == 'transonic':
            self.xmin_nozzle = 0  # may differ from self.x_min if normalizing
            self.xmax_nozzle = 10 # may differ from self.x_max if normalizing
            self.T01 = 300         # Temperature in Kelvin at inlet
            self.p01 = 100*1000   # Total Pressure in Pa at inlet
            self.s_crit = 1.0     # critical nozzle area. mach=1 when svec=s_crit
            self.shockx = 7.      # x location of shock  *not valid for all nozzle shapes*
            self.transx = 5.      # x location of transition, svec=s_crit *not valid for all nozzle shapes*
            self.k2 = 0.5         # coefficient for first order dissipation, usually 0.5
            self.k4 = 0.02        # coefficient for third order dissipation, usually 0.02
            self.isperiodic = False
            self.steady = True
            
        elif self.test_case == 'shock tube':
            self.xmin_nozzle = 0  # may differ from self.x_min if normalizing
            self.xmax_nozzle = 10 # may differ from self.x_max if normalizing
            self.xmembrane = 5    # membrane x point - for shock tube
            self.pL = 1e5         # initial pressire in Pa of left state
            self.rhoL = 1         # initial density in Kg/m^3 of left state
            self.pR = 1e4         # initial pressire in Pa of right state
            self.rhoR = 0.125     # initial density in Kg/m^3 of right state
            self.t_final = 0.0061 # final time to run to, usually 0.0061. Set t_final=None in solver.
            self.k2 = 0.5         # coefficient for first order FD dissipation, usually 0.5
            self.k4 = 0.02        # coefficient for third order FD dissipation, usually 0.02
            self.nozzle_shape = 'constant'
            self.isperiodic = False
            self.steady = False
            
        elif self.test_case == 'density wave':
            self.xmin_nozzle = -1 # may differ from self.x_min if normalizing
            self.xmax_nozzle = 1  # may differ from self.x_max if normalizing
            self.k2 = 0.          # coefficient for first order FD dissipation, usually 0
            self.k4 = 0.02        # coefficient for third order FD dissipation, usually 0.02
            self.u0 = 0.1         # initial (ideally constant) velocity
            self.p0 = 10          # initial (ideally constant) pressure
            self.nozzle_shape = 'constant'
            self.q0_type = 'density wave'
            self.isperiodic = True
            self.norm_var = False
            self.steady = False
            
        else: raise Exception('Test case not understood.')

        self.length_nozzle = self.xmax_nozzle - self.xmin_nozzle

        if self.isperiodic == False:
            
            ''' Calculate the exact solution at the boundary to apply the BC '''
            xx_bdy = np.array([self.xmin_nozzle, self.xmax_nozzle])
            q_exa_bdy = self.exact_sol(xx=xx_bdy, reshape=False, normalize=False)
            self.qL_unnorm = q_exa_bdy[:3]
            self.qR_unnorm = q_exa_bdy[-3:]
    
            ''' Parameters at the boundaries '''   
            self.s_at_qL = self.fun_s(np.array([self.xmin_nozzle]))[0]
            self.s_at_qR = self.fun_s(np.array([self.xmax_nozzle]))[0]
    
            if self.norm_var:            
                if self.norm_loc == 'left':
                    self.q_ref_norm = self.qL_unnorm
                    self.s_at_q_norm = self.s_at_qL
                elif self.norm_loc == 'right':
                    self.q_ref_norm = self.qR_unnorm
                    self.s_at_q_norm = self.s_at_qR
                else:
                    raise Exception('Normalization location is not valid')
        
                self.len_norm = self.length_nozzle
                self.qL = self.normalize_q(self.qL_unnorm)
                self.qR = self.normalize_q(self.qR_unnorm)
        
                self.s_at_qL /= self.len_norm
                self.s_at_qR /= self.len_norm
            else:
                self.qL = self.qL_unnorm
                self.qR = self.qR_unnorm
                
            self.rhoL, self.uL, self.eL, self.PL, self.aL = self.cons2prim(self.qL, self.s_at_qL)
            self.rhoR, self.uR, self.eR, self.PR, self.aR = self.cons2prim(self.qR, self.s_at_qR)
            
            self.EL = self.calcE(self.qL)
            self.ER = self.calcE(self.qR)
                
        else:
            assert(self.norm_var == False),'Normalizing for periodic solutions not set up'

    def fun_s(self, xvec_in):
        ''' Defines the shape of the nozzle. '''
        
        if xvec_in.ndim == 1:
            xvec = xvec_in
        elif xvec_in.ndim == 2:
            # if given x,y,z coordinate format, simply take x values
            xvec = xvec_in[:,0]
        else: raise Exception('Inputted x array not correct.')
        
        # Nozzle from Hirsch and Zingg
        def fun_s_book(xvec):
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

    def fun_der_s(self, xvec_in):
        ''' Defines the derivative of the nozzle. '''
        
        if xvec_in.ndim == 1:
            xvec = xvec_in
        elif xvec_in.ndim == 2:
            # if given x,y,z coordinate format, simply take x values
            xvec = xvec_in[:,0]
        else: raise Exception('Inputted x array not correct.')
        
        # Nozzle from Hirsch and Zingg
        def fun_der_s_book(xvec):
            svec = -(3/5)*(1-xvec/5)
            idx = np.argmax(xvec>5)
            if len(xvec)>1:
                svec[idx:] = -(1/5)*(1-xvec[idx:]/5)
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

    """    
    def calc_p(self,rho,rhou,e):
        ''' function to calculate the pressure given Q variables '''
        # Note: If fed in variables Q*S instead of Q, will return p*S instead of p, as desired
        return (self.g-1)*(e-(rhou**2/2/rho))

    def calc_a(self,rho,rhou,e):
        ''' function to calculate the sound speed given pressure and Q1 '''
        # Note: Regardless if fed Q*S or Q, will always return a, not a*S
        return np.sqrt(self.g*self.calc_p(rho,rhou,e)/rho)
    """    
    def normalize_q(self, q, q_norm=[], s_at_q_norm=0, len_norm=0):
        ''' normalize q according to reference q_norm, where q includes s
        dependance. Normalize such that rho of normalized q is 1, and domain
        goes from 0 to 1. q is a vector state whereas s_at_q_norm is float'''
        
        if len(q_norm) == 0: q_norm = self.q_ref_norm
        if s_at_q_norm == 0: s_at_q_norm = self.s_at_q_norm
        if len_norm == 0: len_norm = self.len_norm

        rho_norm, _, _, p_norm, a_norm = self.cons2prim(q_norm, s_at_q_norm)

        q_0, q_1, q_2 = np.copy(self.decompose_q(q))

        q_0 /= rho_norm * len_norm
        q_1 /= rho_norm * a_norm * len_norm
        q_2 /= rho_norm * a_norm**2 * len_norm

        q_normalized = self.assemble_vec((q_0, q_1, q_2))
        return q_normalized

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

    def exact_sol(self, time=0, xx=[], extra_vars=False, reshape=True, normalize=True):
        ''' Returns the exact solution at given time. Use default time=0 for
        steady solutions. if extra_vars=True, a dictionary with arrays for
        mach, T, p, rho, a, u, and e is also returned along with exa_sol.
        If reshape is True, the solution is returned in shape 
        (self.nen*self.neq_node,self.nelem). If normalize is True and self.norm_var
        is also True, the returned solution will be normalized'''

        assert self.dim==1, 'exact sol only setup for 1D'

        ntol=1e-14 # tolerance for numerical solution of equations
        
        if len(xx)==0: # if no xx given, use unnormed mesh (will norm later if needed)
            if self.xy_not_norm.ndim == 1:
                xx = self.xy_not_norm
            else:
                xx = self.xy_not_norm[:,0]      
        svec = self.fun_s(xx)
            
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
            
            guess = mach_in*np.ones(len(svec_in))

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
                rhs = (2/(self.g+1)) * (aL/aR) * (1 - (P*pR/pL)**((self.g-1)/2*self.g))
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
            rho, a, u, e = remaining_param(p,T,mach)
            return mach, T, p

        def transonic(svec):
            ''' function to solve the transonic flow exactly.
            Requires:   xx = flattened array of node locations in x
                        self.shockx = x location of shock
                        svec, self.s_crit = nozzle shape and critical area
                        self.R , self.g = flow variables
                        T01 , p01 = total temp and pressure at the inlet '''
            # get index for shock location, shockx. If is not exact, add it and delete later.
            shocki = np.where((abs(xx-self.shockx))<1e-10)[0]
            # get index for transition to supersonic flow (mach number = 1, or svec=s_crit).
            transi = np.where((abs(xx-self.transx))<1e-10)[0]
            
            ''' check returned indices. If no indices were given, svec must be
                modidified by adding a node at the location, then later deleted.
                If a unique node exists there, treat it as if it were on the LHS.
                If 2 nodes exist there, treat one on either side of the location.
                If >2 nodes esist there, raise an Exception. '''
            del_shocki = False
            if shocki.size == 0:
                shocki = np.where(xx>self.shockx)[0][0] + 1 # +1 because of slicing later on 
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
                transi = np.where(xx>self.transx)[0][0] + 1 # +1 because of slicing later on 
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
            mach = np.empty(len(xx))
            p = np.empty(len(xx))
            rho = np.empty(len(xx))
            a = np.empty(len(xx))
            u = np.empty(len(xx))
        
            for i in range(len(xx)):
                if xx[i] < xL5 + 1e-10: # region L
                    rho[i] = self.rhoL
                    mach[i] = 0
                    p[i] = self.pL
                    #a[i] = np.sqrt(self.g*self.pL/self.rhoL)
                    #u[i] = 0
                elif xx[i] < x53 + 1e-10: # get pressure and density in region 5 (expansion fan)
                    u[i] = 2*((xx[i]-self.xmembrane)/tf+aL)/(self.g+1)
                    a[i] = u[i] - (xx[i]-self.xmembrane)/tf
                    p[i] = self.pL*(a[i]/aL)**(2*self.g/(self.g-1))
                    rho[i] = self.g*p[i]/a[i]**2
                    mach[i] = u[i]/a[i]
                elif xx[i] < x32 + 1e-10: # region 3 (between expansion fan and contact surface)
                    rho[i] = rho3
                    p[i] = p3
                    a[i] = np.sqrt(self.g*p3/rho3)
                    mach[i] = V/a[i]
                    #u[i] = V
                elif xx[i] < x2R + 1e-10: # region 2 (between contact surface and shock)
                    rho[i] = rho2
                    p[i] = p2
                    a[i] = np.sqrt(self.g*p2/rho2)
                    mach[i] = V/a[i]
                    #u[i] = V
                elif xx[i] < xx[-1] + 1e-10: # region R
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
            assert(self.isperiodic),'Must be periodic domain'
            assert(self.q0_type == 'density wave'),'Initial condition must be density wave'
            
            # just like linear convection equation. See Gassner et al 2020
            xy_mod = np.mod((xx - self.xmin_nozzle) - self.u0*tf, self.length_nozzle) + self.xmin_nozzle
            q = self.set_q0(xy=xy_mod)
            rho, u, e, p, a = self.cons2prim(q.flatten('F'), svec)
            T = p / (rho * self.R)
            mach = u / a
            return mach, T, p

        if self.test_case == 'subsonic':
            mach, T, p = channel_flow(self.T01, self.p01, svec, self.s_crit, 0.3)           
        elif self.test_case == 'transonic':
            mach, T, p = transonic(svec)           
        elif self.test_case == 'shock tube':
            mach, T, p = shocktube(time)
        elif self.test_case == 'density wave':
            mach, T, p = density_wave(time)
            
        rho, a, u, e = remaining_param(p,T,mach)
        exa_sol = self.prim2cons(rho, u, e, svec)
        
        if self.norm_var and normalize: # overwrite if I need to
            exa_sol = self.normalize_q(exa_sol)        
            rho, u, e, p, a = self.cons2prim(exa_sol, svec / self.len_norm)
            T = p / (rho * self.R)
            mach = u / a

        if reshape: exa_sol = np.reshape(exa_sol,(self.nen*self.neq_node,self.nelem),'F')
        
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

        PdeBaseCons.set_mesh(self, mesh)

        if self.norm_var:
            assert self.xmin == 0 and self.xmax == 1, \
                'For solve with normalized solution must have xmin=0 and xmax=1'
            self.xy_not_norm = self.xy * self.length_nozzle + self.xmin_nozzle
            self.xy_elem_not_norm = self.xy_elem * self.length_nozzle + self.xmin_nozzle
        else:
            assert self.xmin == self.xmin_nozzle and self.xmax == self.xmax_nozzle, \
                'For solve without normalized solution must have xmin=xmin_nozzle={0} and xmax=xmax_nozzle={1}'.format(self.xmin_nozzle,self.xmax_nozzle)
            self.xy_not_norm = self.xy
            self.xy_elem_not_norm = self.xy_elem

        # Calculate the shape of the nozzle at the mesh and boundary nodes
        self.svec_not_norm = self.fun_s(self.xy_not_norm)
        self.svec_der = self.fun_der_s(self.xy_not_norm) # does not need to be normalized (unitless)

        if self.norm_var:
            self.svec = self.svec_not_norm / self.len_norm
        else:
            self.svec = self.svec_not_norm
            
        self.svec_elem = np.reshape(self.svec,(self.nen,self.nelem),'F')
        self.svec_der_elem = np.reshape(self.svec_der,(self.nen,self.nelem),'F')
            

    def set_q0(self, xy=[]):
        
        if len(xy)==0:
            xy = self.xy_not_norm[:,0]
            svec = self.svec_not_norm
        else:
            svec = self.fun_s(xy)

        if self.test_case == 'density wave':
            self.q0_type == 'exact'
            rho = 1 + 0.98*np.sin(2*np.pi*xy)
            u = self.u0 * np.ones(rho.shape)
            p = self.p0 * np.ones(rho.shape)
            e = p/(self.g-1) + rho * u**2 /2
            q0 = self.prim2cons(rho, u, e, svec)
            if self.norm_var:
                q0 = self.normalize_q(q0)
                
        elif self.test_case == 'shock tube':
            self.q0_type == 'exact'
            q0 = self.exact_sol() # also normalizes if needed
            
        else: # subsonic or transonic
            if self.q0_type == 'exact':
                q0 = self.exact_sol() # also normalizes if needed
    
            elif self.q0_type == 'linear':
                assert(self.isperiodic == False),'Using boundary values to initialize. Must be non-periodic.'
                rho = np.linspace(self.rhoL, self.rhoR, self.nn)
                u = np.linspace(self.uL, self.uR, self.nn)
                e = np.linspace(self.eL, self.eR, self.nn)
                q0 = self.prim2cons(rho, u, e, svec)
                if self.norm_var:
                    q0 = self.normalize_q(q0)
            else:
                raise Exception('Unknown init sol type')

        # restructure in shape (nen,nelem), i.e. columns are each element
        return np.reshape(q0,(self.nen*self.neq_node,self.nelem),'F')

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
        elif self.plt_var2plot_name == 'mach':
            return u / a
        else:
            raise Exception('Requested variable to plot is not available')

    def calcE(self, q):

        q_0, q_1, q_2 = self.decompose_q(q)
        u = q_1 / q_0

        k = u*q_1                    # Common term: rho * u^2 * S
        ps = (self.g-1)*(q_2 - k/2)  # p * S

        e0 = q_1
        e1 = k + ps
        e2 = u*(q_2 + ps)

        E = self.assemble_vec((e0, e1, e2))
        return E
    
    def dEdq(self, q):
        q_0, q_1, q_2 = self.decompose_q(q)
        u = q_1 / q_0

        '''
        n_node = q_0.shape[0]
        # dEdq is complex if using complex step for implicit time
        # marching with SBP operators
        if np.any(np.iscomplex(q)):
            dEdq = np.zeros((n_node, 3, 3), dtype=complex)
        else:
            dEdq = np.zeros((n_node, 3, 3))
        '''
        
        u2 = u**2
        q2_q0 = q_2/q_0

        # entries of the dEdq (A) matrix
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
    
    def calcG(self, q, elem_idx=None):

        q_0, q_1, q_2 = self.decompose_q(q)
        
        if elem_idx == None:
            svec_der = self.svec_der_elem
            svec = self.svec_elem
        else:
            svec_der = self.svec_der_elem[:,elem_idx]
            svec = self.svec_elem[:,elem_idx]

        p = (self.g-1)*(q_2 - 0.5*q_1**2 /q_0) /svec
        g2 = p * svec_der
        zero_vec = np.zeros((self.nen,self.nelem))
        G = self.assemble_vec((zero_vec, g2, zero_vec))
        return G
    '''
    def dGdq(self, q=None, xy_idx0=None, xy_idx1=None):
        # TODO: Not sure about shape

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
    '''

class Quasi1dEulerSbp(SatBaseCons, Quasi1dEuler):

    def dEdq_eig_abs(self, dEdq):
        #TODO: set this up as an imported function. Then maybe make functions a class
        
        if dEdq.ndim == 2:
            eig_val, eig_vec = np.linalg.eig(dEdq)
            dEdq_eig_abs = eig_vec @ np.diag(np.abs(eig_val)) @ np.linalg.inv(eig_vec)
        elif dEdq.ndim ==3:
            dEdq_mod = np.transpose(dEdq,axes=(2,0,1))
            eig_val, eig_vec = np.linalg.eig(dEdq_mod)
            dEdq_eig_abs = np.zeros(dEdq.shape)
            for i in range(len(eig_val)):
                dEdq_eig_abs[:,:,i] = eig_vec[i,:,:] @ np.diag(np.abs(eig_val[i])) @ np.linalg.inv(eig_vec[i,:,:])

        return dEdq_eig_abs
    '''
    def dfdq(self, q,  xy_idx0=None, xy_idx1=None):
        A = self.dEdq(q)
        dGdq = self.dGdq(q, xy_idx0, xy_idx1)

        dfdq = -self.der1 @ A + dGdq
        return dfdq    
    '''