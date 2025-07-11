#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 23:51:27 2021

@author: bercik
"""

import numpy as np

from Source.DiffEq.DiffEqBase import PdeBase
import Source.Methods.Functions as fn
import Source.DiffEq.EulerFunctions as efn

class Euler(PdeBase):
    
    # Equation constants
    diffeq_name = 'Euler2d'
    dim = 2
    neq_node = 4            # No. of equations per node
    pde_order1 = True
    pde_order2 = False
    has_exa_sol = True
    para_names = (r'$R$',r'$\gamma$',)
    enforce_positivity = True
    t_scale = 1.
    a_inf = 1.
    rho_inf = 1.

    # Problem constants
    R_fix = 287
    g_fix = 1.4
    para_fix = [R_fix,g_fix]

    # Plotting constants
    var2plot_name = r'$\rho$' # rho, u, e, p, a, mach    

    def __init__(self, para, q0_type=None, test_case='subsonic_nozzle',bc='periodic', nondimensionalize=True):

        ''' Add inputs to the class '''

        super().__init__(para, q0_type)
        self.R = para[0]
        self.g = para[1]
        self.test_case = test_case
        self.bc = bc
        
        if self.q0_type == None:
            self.q0_type = 'exact' # can be exact, ?

        self.nondimensionalize = nondimensionalize
        if self.nondimensionalize:
            print('Using non-dimensionalized variables.')

        ''' Set flow and problem dependent parameters  '''
        
        if 'density_wave' in self.test_case:
            self.xmin_fix = (-1.,-1.)  # should be the same as self.x_min
            self.xmax_fix = (1.,1.) # should be the same as self.x_max
            if '1dx' in self.test_case:
                self.u0 = 0.1        # initial (ideally constant) velocity
                self.v0 = 0.0
            elif '1dy' in self.test_case:
                self.u0 = 0.0        # initial (ideally constant) velocity
                self.v0 = 0.1
            else:
                self.u0 = 0.1        # initial (ideally constant) velocity
                self.v0 = 0.2
            self.p0 = 20          # initial (ideally constant) pressure
            if self.q0_type != 'density_wave':
                print("WARNING: Overwriting inputted q0_type to 'density_wave'.")
                self.q0_type = 'density_wave'
            assert (bc == 'periodic'),\
                "density_wave must use bc='periodic'."
            self.steady = False

            if self.nondimensionalize:
                self.rho_inf = 1.
                self.a_inf = np.sqrt(self.g*self.p0/self.rho_inf)
                self.e_inf = self.rho_inf * self.a_inf * self.a_inf 
                self.rhou_inf = self.rho_inf * self.a_inf
                self.t_scale = self.a_inf

        elif self.test_case == 'vortex' or self.test_case == 'vortex_lowma':
            self.xmin_fix = (-5.,-5.)  # should be the same as self.x_min
            self.xmax_fix = (5.,5.) # should be the same as self.x_max
            if self.test_case == 'vortex_lowma':
                self.mach0 = 0.1
            else:
                self.mach0 = 0.5
            self.beta = 0.2
            self.vtx_rad = 0.5
            self.a_inf = 1.
            self.u0 = self.mach0 * self.a_inf
            self.v0 = 0.0
            if self.q0_type != 'vortex':
                print("WARNING: Overwriting inputted q0_type to 'vortex'.")
                self.q0_type = 'vortex'
            assert (bc == 'periodic'),\
                "vortex must use bc='periodic'."
            self.steady = False

            if self.nondimensionalize:
                self.rho_inf = 1.
                self.e_inf = self.rho_inf * self.a_inf * self.a_inf 
                self.rhou_inf = self.rho_inf * self.a_inf
                self.t_scale = self.a_inf

        elif self.test_case == 'kelvin-helmholtz' or self.test_case == 'kelvin-helmholtz_asym' or self.test_case == 'kelvin-helmholtz_noperturb':
            if self.test_case == 'kelvin-helmholtz_noperturb':
                self.has_exa_sol = True
            else:
                self.has_exa_sol = False
            self.xmin_fix = (-1.,-1.)  # should be the same as self.x_min
            self.xmax_fix = (1.,1.) # should be the same as self.x_max
            if self.q0_type != 'kelvin-helmholtz':
                print("WARNING: Overwriting inputted q0_type to 'kelvin-helmholtz'.")
                self.q0_type = 'kelvin-helmholtz'
            assert (bc == 'periodic'),\
                "kelvin-helmholtz must use bc='periodic'."
            self.steady = False
            
            if self.nondimensionalize:
                self.rho_inf = 0.5
                p_inf = 1.
                a_inf = np.sqrt(self.g*p_inf/self.rho_inf)
                self.rhou_inf = self.rho_inf*a_inf
                self.e_inf = self.rho_inf*a_inf*a_inf
                self.t_scale = a_inf

        elif self.test_case == 'taylor-green':
            self.has_exa_sol = False
            self.xmin_fix = (-np.pi,-np.pi)  # should be the same as self.x_min
            self.xmax_fix = (np.pi,np.pi) # should be the same as self.x_max
            if self.q0_type != 'taylor-green':
                print("WARNING: Overwriting inputted q0_type to 'taylor-green'.")
                self.q0_type = 'taylor-green'
            assert (bc == 'periodic'),\
                "taylor-green must use bc='periodic'."
            self.steady = False
            self.mach0 = 0.1
            
            if self.nondimensionalize:
                self.rho_inf = 1.0
                p_inf = self.rho_inf/(self.mach0*self.mach0*self.g) + self.rho_inf*(3./16.)
                a_inf = np.sqrt(self.g*p_inf/self.rho_inf)
                self.rhou_inf = self.rho_inf*a_inf
                self.e_inf = self.rho_inf*a_inf*a_inf
                self.t_scale = a_inf


        elif self.test_case == 'manufactured_soln':

            raise Exception('Manufactured solution not yet implemented.')
        
            self.xmin_fix = 0.  # should be the same as self.x_min
            self.xmax_fix = 2. # should be the same as self.x_max
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
            
        else: raise Exception("Test case not understood. Try 'vortex', 'manufactured_soln', 'density_wave', or 'density_wave_1d.")
        
        if self.g == self.g_fix:
            print('Using the fixed g={0} diffeq functions since params match.'.format(self.g_fix))
            self.calcEx = efn.calcEx_2D
            self.calcEy = efn.calcEy_2D
            self.calcExEy = efn.calcExEy_2D
            self.dExdq = efn.dExdq_2D
            self.dEydq = efn.dEydq_2D
            self.dEndq = efn.dEndq_2D
            self.dEndq_abs_dq = efn.dEndq_abs_dq_2D
            self.central_fluxes = efn.Central_fluxes_2D
            self.ismail_roe_fluxes = efn.Ismail_Roe_fluxes_2D
            self.ranocha_fluxes = efn.Ranocha_fluxes_2D
            self.chandrashekar_fluxes = efn.Chandrashekar_fluxes_2D
            self.maxeig_dExdq = efn.maxeig_dExdq_2D
            self.maxeig_dEydq = efn.maxeig_dEydq_2D
            self.maxeig_dEndq = efn.maxeig_dEndq_2D
            self.entropy = efn.entropy_2D
            self.entropy_var = efn.entropy_var_2D
            self.dqdw = efn.symmetrizer_2D
            self.dEndq_abs = efn.dEndq_abs_2D
            self.dEndw_abs = efn.dEndw_abs_2D
            self.dEndq_abs_derigs = efn.dEndq_abs_2D_derigs
            self.dEndw_abs_derigs = efn.dEndw_abs_2D_derigs
            self.dqdw_derigs = efn.symmetrizer_dw_derigs_2D
            self.calc_p = efn.calc_p_2D
            self.roe_avg = efn.Roe_avg_2D
            self.ismail_roe_avg = efn.Ismail_Roe_avg_2D
            self.derigs_avg = efn.Derigs_avg_2D


        if bc != 'periodic':
            
            raise Exception('Only periodic BCs are currently supported.')



    ######################################
    ''' Begin  defining class functions'''
    ######################################
      
    def calc_p(self,q):
        ''' function to calculate the pressure given q '''
        rho = q[::4,:]
        rhou = q[1::4,:]
        rhov = q[2::4,:]
        e = q[3::4,:]
        p = (self.g-1)*(e-0.5*(rhou*rhou + rhov*rhov)/rho)
        return p

    def calc_a(self,q):
        ''' function to calculate the sound speed '''
        rho = q[::4,:]
        return np.sqrt(self.g*self.calc_p(q)/rho)   

    def check_positivity(self, q):
        ''' Check if thermodynamic variables are positive '''
        rho = q[::4,:]
        p = self.calc_p(q)
        not_ok = np.any(rho < 0) or np.any(p < 0)
        return not_ok  

    def decompose_q(self, q):
        ''' splits q[nen*neq_node,nelem] or q[nen*neq_node] to q_i[nen,nelem] 
        or q_i[nen] for each i in neq_node ''' 

        assert self.neq_node == 4, 'decompose_q only works for neq_node=4'

        q_0 = q[0::4]
        q_1 = q[1::4]
        q_2 = q[2::4]
        q_3 = q[3::4]

        return q_0, q_1, q_2, q_3

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
    def prim2cons(rho, u, v, e):
        '''takes y_i[nen,nelem] or y_i[nen] primitive variables (for each i) to 
        q[nen*neq_node,nelem] or q[nen*neq_node] conservative vector, as long 
        as svec is in a shape that matches y_i '''
        
        q_0 = rho
        q_1 = q_0 * u
        q_2 = q_0 * v
        q_3 = e 

        q = Euler.assemble_vec((q_0, q_1, q_2, q_3))

        return q

    def cons2prim(self, q):
        '''takes q[nen*neq_node,nelem] or q[nen*neq_node] conservative vector 
        to y_i[nen,nelem] or y_i[nen] primitive variables (for each i) '''

        rho, q_1, q_2, e = self.decompose_q(q)

        #rho = q_0
        u = q_1 / rho
        v = q_2 / rho
        #e = q_3
        P = (self.g-1)*(e - (rho * (u*u + v*v)/2))

        a = np.sqrt(self.g * P/rho)

        # Convert numpy array with 1 node to scalar values
        if q.size == 4 and q.ndim==1:
            rho = rho[0]
            u = u[0]
            v = v[0]
            e = e[0]
            P = P[0]
            a = a[0]
        return rho, u, v, e, P, a
    
    def entropy(self, q):
        ''' return the nodal values of the entropy s(q). '''
        # Note: this is not quite the "normal" entropy for quasi1D euler when \neq 1, but is a correct entropy
        rho, q_1, q_2, e = self.decompose_q(q)
        u = q_1 / rho
        v = q_2 / rho
        p = (self.g-1)*(q_2 - rho*0.5*(u*u+v*v))
        s = np.log(p/(rho**self.g))
        return (-rho*s/(self.g-1))
    
    def entropy_var(self, q):
        ''' return the nodal values of the entropy variables w(q). '''
        # Note: the same entropy variables for quasi1D euler (no svec dependence)
        rho, q_1, q_2, e = self.decompose_q(q)
        u = q_1 / rho
        v = q_2 / rho
        p = (self.g-1)*(q_2 - rho*0.5*(u*u+v*v))
        s = np.log(p/(rho**self.g))
        w = self.assemble_vec(((self.g-s)/(self.g-1) - 0.5*rho*(u*u+v*v)/p, rho*u/p, rho*v/p, -rho/p))
        return w
    
    def kinetic_energy(self, q):
        ''' return the kinetic energy at each node. '''
        rho, q_1, q_2, _ = self.decompose_q(q)
        u = q_1 / rho
        v = q_2 / rho
        return 0.5*rho*(u*u + v*v)
    
    def enstrophy(self, q):
        ''' return the enstrophy at each node. '''
        rho, q_1, q_2, _ = self.decompose_q(q)
        u = q_1 / rho
        v = q_2 / rho
        w = self.gm_gv(self.Dx,v) - self.gm_gv(self.Dy,u)
        return 0.5*rho*(w*w)

    def exact_sol(self, time=0, xy=None, extra_vars=False, nondimensionalize=None, guess=None):
        ''' Returns the exact solution at given time. Use default time=0 for
        steady solutions. if extra_vars=True, a dictionary with arrays for
        mach, T, p, rho, a, u, and e is also returned along with exa_sol.  '''

        if nondimensionalize is None:
            nondimensionalize = self.nondimensionalize

        if self.nondimensionalize:
            time /= self.t_scale
        
        if xy is None:
            xy = self.xy_elem   
        
        def density_wave(tf):
            ''' function to solve a simple travelling wave. '''
            
            if self.q0_type != 'density_wave':
                print("ERROR: for exact_sol, initial condition must be density_wave, not '"+self.q0_type+"'")
                return np.zeros(np.shape(xy)), np.zeros(np.shape(xy)), np.zeros(np.shape(xy)), np.zeros(np.shape(xy))
            
            # just like linear convection equation. See Gassner et al 2020
            xy_mod = np.empty(xy.shape)
            xy_mod[:,0,:] = np.mod((xy[:,0,:] - self.xmin[0]) - self.u0*tf, self.dom_len[0]) + self.xmin[0]
            xy_mod[:,1,:] = np.mod((xy[:,1,:] - self.xmin[1]) - self.v0*tf, self.dom_len[1]) + self.xmin[1]
            q = self.set_q0(xy=xy_mod)

            return q
        
        def vortex(tf):
            ''' function to solve a simple travelling isentropic vortex. '''
            
            if self.q0_type != 'vortex':
                print("ERROR: for exact_sol, initial condition must be vortex, not '"+self.q0_type+"'")
                return np.zeros(np.shape(xy)), np.zeros(np.shape(xy)), np.zeros(np.shape(xy)), np.zeros(np.shape(xy))
            
            # just like linear convection equation.
            xy_mod = np.empty(xy.shape)
            xy_mod[:,0,:] = np.mod((xy[:,0,:] - self.xmin[0]) - self.u0*tf, self.dom_len[0]) + self.xmin[0]
            xy_mod[:,1,:] = np.mod((xy[:,1,:] - self.xmin[1]) - self.v0*tf, self.dom_len[1]) + self.xmin[1]
            q = self.set_q0(xy=xy_mod)

            return q
        
        def kelvin_helmholtz(tf):
            ''' initial condition for kelvin-helmholtz problem '''

            if self.q0_type != 'kelvin-helmholtz':
                print("ERROR: for exact_sol, initial condition must be kelvin-helmholtz, not '"+self.q0_type+"'")
            
            q = self.set_q0(xy=xy)
            return q

        
        def manufactured_soln(t):
            ''' manufactured solution. '''

            raise Exception('Not implemented yet.')
            
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
        
        def remaining_param(q):
            ''' calculate density, sound speed, velocity, and energy once
            total pressure, temperature, and mach number are known '''
            rho, u, v, e, P, a = self.cons2prim(q)
            s = self.entropy(q)
            mach = np.sqrt(u*u + v*v) / a
            T = P / (rho * self.R)
            return rho, u, v, e, P, a, s, mach, T

        if 'density_wave' in self.test_case:
            exa_sol = density_wave(time)
        elif self.test_case == 'vortex' or self.test_case == 'vortex_lowma':    
            exa_sol = vortex(time)
        elif self.test_case == 'manufactured_soln':
            exa_sol = manufactured_soln(time)
        elif self.test_case == self.test_case == 'kelvin-helmholtz_noperturb':
            exa_sol = kelvin_helmholtz(time)
        else:
            raise Exception('Invalid test case.')
            
        if nondimensionalize:
            exa_sol[0::4] = exa_sol[0::4] / self.rho_inf
            exa_sol[1::4] = exa_sol[1::4] / self.rhou_inf
            exa_sol[2::4] = exa_sol[2::4] / self.rhou_inf
            exa_sol[0::4] = exa_sol[0::4] / self.e_inf
        rho, u, v, e, p, a, s, mach, T = remaining_param(exa_sol)

        
        if extra_vars:
            var_names = ['mach', 'T', 'p', 'rho', 'a', 'u', 'v', 'e', 's']
            exa_sol_extra = {}
            for i in range(len(var_names)):
                values = eval(var_names[i])
                exa_sol_extra[var_names[i]] = values 
            return exa_sol, exa_sol_extra
        else: return exa_sol
            

    def set_q0(self, q0_type=None, xy=None):
        # overwrite base function from PdeBase
        
        if q0_type is None:
            q0_type = self.q0_type
        q0_type = q0_type.lower()

        if xy is None:
            xy = self.xy_elem

        if 'density_wave' in self.test_case:
            if q0_type != 'density_wave':
                print("WARNING: Instead of using q0_type = '"+q0_type+", you should probably use q0_type = 'density_wave'.")
                q0 = PdeBase.set_q0(self, q0_type=q0_type, xy=xy)
                fn.repeat_neq_gv(q0,self.neq_node)
            else:
                if self.test_case == 'density_wave':
                    rho = 1 + 0.98*np.sin(2*np.pi*(xy[:,0,:]+xy[:,1,:]))
                elif self.test_case == 'density_wave_1dx':
                    rho = 1 + 0.98*np.sin(2*np.pi*xy[:,0,:])
                elif self.test_case == 'density_wave_1dy':
                    rho = 1 + 0.98*np.sin(2*np.pi*xy[:,1,:])
                else:
                    raise Exception('Invalid test case.')
                u = self.u0 * np.ones(rho.shape)
                v = self.v0 * np.ones(rho.shape)
                p = self.p0 * np.ones(rho.shape)
                e = p/(self.g-1) + 0.5 * rho * (u*u + v*v)
                q0 = self.prim2cons(rho, u, v, e)

        elif self.test_case == 'manufactured_soln':
            assert(q0_type == 'manufactured_soln'),"Must use q0_type == 'manufactured_soln'."
            rho = 2 + 0.1*np.sin(np.pi*(xy[:,0,:]+xy[:,1,:]))
            u = np.ones(rho.shape)
            v = np.ones(rho.shape)
            e = rho*rho
            p = (self.g-1)*rho*(rho-0.5)
            q0 = self.prim2cons(rho, u, v, e)
                
        elif self.test_case == 'vortex' or self.test_case == 'vortex_lowma':
            if q0_type != 'vortex':
                print('WARNING: Ignoring q0_type and setting initial conditions to vortex.')
            r2 = - (xy[:,0,:]/self.vtx_rad)**2 - (xy[:,1,:]/self.vtx_rad)**2
            u = self.mach0 - self.mach0*self.beta*xy[:,1,:]/self.vtx_rad*np.exp(0.5*r2)
            v = self.mach0*self.beta*xy[:,0,:]/self.vtx_rad*np.exp(0.5*r2)
            rho = (1. - 0.5*(self.mach0*self.beta)**2*(self.g-1)*np.exp(r2))**(1./(self.g-1))
            p = (rho**self.g)/self.g
            e = p/(self.g-1) + 0.5 * rho * (u*u + v*v)
            q0 = self.prim2cons(rho, u, v, e)

        elif self.test_case == 'kelvin-helmholtz' or self.test_case == 'kelvin-helmholtz_asym' or self.test_case == 'kelvin-helmholtz_noperturb':
            if q0_type != 'kelvin-helmholtz':
                print('WARNING: Ignoring q0_type and setting initial conditions to kelvin-helmholtz.')
            B = np.tanh( 15 * xy[:,1,:] + 7.5 ) - np.tanh( 15 * xy[:,1,:] - 7.5 )
            rho = 0.5 + 0.75 * B
            u = 0.5 * ( B - 1 )
            if self.test_case == 'kelvin-helmholtz_asym':
                v = 0.1 * np.sin( 2 * np.pi * xy[:,0,:] ) * ( 1. + 0.01 * np.sin( np.pi * xy[:,0,:] ) * np.sin( np.pi * xy[:,1,:] ) )
            elif self.test_case == 'kelvin-helmholtz_noperturb':
                v = np.zeros_like(xy[:,0,:])
            else:
                v = 0.1 * np.sin( 2 * np.pi * xy[:,0,:] )
            e = 1 / ( self.g - 1 ) + 0.5 * rho * (u*u + v*v)
            q0 = self.prim2cons(rho, u, v, e)

        elif self.test_case == 'taylor-green':
            if q0_type != 'taylor-green':
                print('WARNING: Ignoring q0_type and setting initial conditions to taylor-green.')
            rho = np.ones_like(xy[:,0,:])
            u = np.sin(xy[:,0,:]) * np.cos(xy[:,1,:])
            v = -np.cos(xy[:,0,:]) * np.sin(xy[:,1,:])
            p = rho/(self.mach0*self.mach0*self.g) + rho*(np.cos(2*xy[:,0,:]) + np.cos(2*xy[:,1,:]))*(3./16.)
            e = p/(self.g-1) + 0.5 * rho * (u*u + v*v)
            q0 = self.prim2cons(rho, u, v, e)
            
        else: 
            print("WARNING: Instead of using q0_type = '"+q0_type+"', you should probably use q0_type = "+self.test_case+".")
            q0 = PdeBase.set_q0(self, q0_type=q0_type, xy=xy)
            fn.repeat_neq_gv(q0,self.neq_node)
        
        if self.nondimensionalize:
            q0[0::4] = q0[0::4] / self.rho_inf
            q0[1::4] = q0[1::4] / self.rhou_inf
            q0[2::4] = q0[2::4] / self.rhou_inf
            q0[3::4] = q0[3::4] / self.e_inf

        return q0


    def var2plot(self, q_in, var2plot_name):

        q = np.copy(q_in)
        if self.nondimensionalize:
            # re-dimensionalize
            q[0::4] = q[0::4] * self.rho_inf
            q[1::4] = q[1::4] * self.rhou_inf
            q[2::4] = q[2::4] * self.rhou_inf
            q[3::4] = q[3::4] * self.e_inf
        
        if var2plot_name is None:
            var2plot_name = self.var2plot_name
        rho, u, v, e, P, a = self.cons2prim(q)
        #T = P / (rho * self.R)
        #w = self.entropy_var(q)

        if var2plot_name == 'rho' or var2plot_name == r'$\rho$':
            return rho
        elif var2plot_name == 'u' or var2plot_name == r'$u$':
            return u
        elif var2plot_name == 'v' or var2plot_name == r'$v$':
            return v
        elif var2plot_name == 'rhou' or var2plot_name == r'$\rho u$':
            return rho*u
        elif var2plot_name == 'rhov' or var2plot_name == r'$\rho v$':
            return rho*v
        elif var2plot_name == 'e' or var2plot_name == r'$e$':
            return e
        elif var2plot_name == 'p' or var2plot_name == r'$p$':
            return P
        elif var2plot_name == 'a' or var2plot_name == r'$a$':
            return a
        elif var2plot_name == 'mach' or var2plot_name == r'$M$' or var2plot_name == 'Ma':
            return np.sqrt(u*u + v*v) / a
        elif var2plot_name == 'w1' or var2plot_name == r'$w_1$':
            w = self.entropy_var(q)
            return w[::3]
        elif var2plot_name == 'w2' or var2plot_name == r'$w_2$':
            w = self.entropy_var(q)
            return w[1::3]
        elif var2plot_name == 'w3' or var2plot_name == r'$w_3$':
            w = self.entropy_var(q)
            return w[2::3]
        elif var2plot_name == 's' or var2plot_name == r'$s$' or var2plot_name == 'entropy':
            return self.entropy(q)
        elif var2plot_name == 'q':
            return q
        else:
            raise Exception('Requested variable to plot is not available, '+var2plot_name)

    def calcEx(self, q):
        return np.zeros_like(q)
    
    def calcEy(self, q):
        return np.zeros_like(q)
    
    def dExdq(self, q):
        return np.zeros((self.nn, 4, 4, self.nelem))
    
    def dEydq(self, q):
        return np.zeros((self.nn, 4, 4, self.nelem))
        
    def dqdw(self,q):
        return np.zeros((self.nn, 4, 4, self.nelem))
    
    def calcG(self, q, t):
        return np.zeros_like(q)
    
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
        dGdq = np.zeros((self.nn * 4, self.nn * 4))
        return dGdq

    def d2Exdq2(self, q):
        return None

    def maxeig_dExdq(self, q):
        ''' return the maximum eigenvalue - used for LF fluxes. accepts q[:,:] and q[:] '''
        rho = q[::4] 
        u = q[1::4]/rho
        v = q[1::4]/rho
        e_rho = q[3::4]/rho 
        p_rho = (self.g-1)*(e_rho-0.5*(u*u+v*v)) # pressure / rho
        a = np.sqrt(self.g*p_rho) # sound speed
        lam = np.maximum(np.abs(u+a),np.abs(u-a))
        return lam
    
    def maxeig_dEydq(self, q):
        ''' return the maximum eigenvalue - used for LF fluxes. accepts q[:,:] and q[:] '''
        rho = q[::4] 
        u = q[1::4]/rho
        v = q[1::4]/rho
        e_rho = q[3::4]/rho 
        p_rho = (self.g-1)*(e_rho-0.5*(u*u+v*v)) # pressure / rho
        a = np.sqrt(self.g*p_rho) # sound speed
        lam = np.maximum(np.abs(v+a),np.abs(v-a))
        return lam
    
    def use_alternative_dEndw_abs(self):
        " overwrite dEndw_abs_2D "
        self.dEndw_abs = efn.dEndw_abs_2D_alternative
        # TODO: do the same for Derigs? (SAT)