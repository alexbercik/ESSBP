#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 12:01:45 2020

@author: bercik
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
from os import path
from socket import gethostname


from Source.TimeMarch.TimeMarching import TimeMarching
from Source.Disc.MakeDgOp import MakeDgOp

class PdeSolver:

    
    # Initialize parameters
    q_sol = None        # Solution of the Diffeq
    cons_obj = None     # Conservation Objective(s)
    keep_all_ts = True  # whether to keep all time steps on solve
    skip_ts = 0
    use_diffeq_dExdx = False
    calc_nd_ops = False # Calulate the physical multi-dimensional operators (incorporates E matrices)
    print_progress = True
    had_flux = 'none'
    tm_atol = None
    tm_rtol = None
    tm_nframes = None
    t_initial = 0.0
    dt = None

    def __init__(self, diffeq, settings,                            # Diffeq
                 tm_method, dt, t_final,                    # Time marching                                  # Initial solution
                 p=2, disc_type='div',                      # Discretization
                 surf_diss=None, vol_diss=None, had_flux='none',
                 nelem=0, nen=0,  disc_nodes='lgl',
                 bc=None, xmin=0, xmax=1,     # Domain
                 cons_obj_name=None,         # Other
                 bool_plot_sol=False, print_sol_norm=False,
                 print_residual=False, check_resid_conv=False,
                 sparse=None, sat_sparse=None):
        '''
        Parameters
        ----------
        diffeq : class
            The class of the differential equation.
        settings : dict
            A bunch of less important optional settings for the discretization
        tm_method : str
            The type of time marching method to use
        dt : float
            Size of the time step.
        t_final : float
            The solution is evaluated up to this point.
        q0 : np array, optional
            If provided, this initial solution is used to start the solution.
            If not provided, the initial solution for the diffeq class is used.
            The default is None.
        p : int, optional
            Degree of the spatial discretization.
            The default is 2.
        disc_type : str, optional
            Indicates the form of spatial discretization that is used, either 
            'div' for divergence or 'had' for hadamard.
        surf_diss : dic or None, optional
            The numerical surface dissipation to use along surface element boudnaries
        vol_diss : dic or None, optional
            The numerical volume dissipation to use
        had_flux : str, optional
            The 2-point numerical flux to use for the Hadamard form
        nelem : int, optional (Not used for FD)
            The number of elements in the mesh.
            The default is 0.
        nen : int, optional (Not used for FD)
            The number of nodes per element. The actual number of nodes per 
            element is set according to whichever of p or nen results in the 
            largest number of nodes (See SbpQuadRule).
            The default is 0.
        disc_nodes : str, optional
            Indicates the type of spatial discretization that is used.
            The default is 'lgl'.
        bc : str or tuple
            Indicates boundary conditions, ex. 'periodic' if the mesh is periodic.
            The default is None, in which case it searches diffeq for default.
        xmin : float or (float,float)
            Min coordinate of the mesh, either x in 1D or (x,y) in 2D
        xmax : float or (float,float)
             Max coordinate of the mesh, either x in 1D or (x,y) in 2D
        xmax : float
             Factor by which to warp the 2D mesh. Typically use 0.2
             The defualt is 0.
        cons_obj_name : str or tuple, optional
            The name or names of the conservation objective(s) to calculate.
            If None, no conservation objectives are calculated.
            The default is None.
        bool_plot_sol : bool, optional
            The solution is plotted at each time step if this is set to True.
            The default is False.
        print_sol_norm : bool, optional
            The norm of the sol is printed at each time step if this is True.
            The default is False.
        print_residual : bool, optional
            The norm of the residual is printed if this flag is true.
            The default is False.
        check_resid_conv : bool, optional
            Whether or not to check for residual convergence to exit.
            The default is False (as it should be for unsteady)
        '''

        ''' Add all inputs to the class '''

        self.diffeq = diffeq
        
        self.settings = settings
        
        ''' Set default settings if not given '''
        self.settings.setdefault('warp_factor',0.) 
        # called by Disc.MakeMesh. Warps / stretches mesh.
        self.settings.setdefault('warp_type','default')
        # called by Disc.MakeMesh. Options: 'defualt', 'papers', 'quad'
        self.settings.setdefault('metric_method','exact') 
        # called by Disc.MakeMesh.get_jac_metrics. Options: 'calculate', 'VinokurYee', 'ThomasLombard', 'exact'
        self.settings.setdefault('bdy_metric_method','exact') 
        # called by Disc.MakeMesh.get_jac_metrics. Options: 'calculate', 'VinokurYee', 'ThomasLombard', 'exact', 'interpolate'
        self.settings.setdefault('jac_method','exact') 
        # called by Disc.MakeMesh.get_jac_metrics. Options: 'calculate', 'match', 'exact', 'deng'
        self.settings.setdefault('use_optz_metrics',True) 
        # called by Disc.MakeMesh.get_jac_metrics. Uses optimized metrics for free stream preservation.
        self.settings.setdefault('calc_exact_metrics',False) 
        # called by Disc.MakeMesh.get_jac_metrics. Calculates exact metrics alongside regular method..
        self.settings.setdefault('metric_optz_method','default') 
        # called by Disc.MakeMesh.get_jac_metrics. Define the metric optimization procedure.
        self.settings.setdefault('had_alpha',1) 
        # Modifies the SAT terms in the Hadamard form. See ESSBP documentation.
        self.settings.setdefault('had_beta',1) 
        # Modifies the SAT terms in the Hadamard form. See ESSBP documentation.
        self.settings.setdefault('had_gamma',1) 
        # Modifies the SAT terms in the Hadamard form. See ESSBP documentation.
        self.settings.setdefault('stop_after_metrics', False) 
        # Do not set up physical operators, SATs, etc. only Mesh setup.
        self.settings.setdefault('stop_after_mesh', False) 
        # Do not set up SATs, etc. only Mesh setup.
        self.settings.setdefault('skew_sym', True)
        # determines whether to use a split-form or divergence form for metrics

        # Time marching
        self.tm_method = tm_method.lower()
        self.t_final = t_final

        # Sparsity
        self.sparse = sparse
        self.sat_sparse = sat_sparse
        
        # Discretization
        self.dim = self.diffeq.dim
        self.disc_nodes = disc_nodes.lower()
        self.neq_node = self.diffeq.neq_node
        if disc_type.lower() == 'div' or disc_type.lower() == 'divergence':
            self.disc_type = 'div'
            self.calc_had_flux = None
            if hasattr(self.diffeq, 'dExdx'):
                print('Using dExdx defined in specific DiffEq file.')
                self.use_diffeq_dExdx = True
                if self.dim>1:
                    #TODO: Update this for 2D and 3D
                    print('TODO: Update use_dEdx for 2D and 3D')
            else:
                if self.dim>1:
                    #TODO: Update this for 2D and 3D
                    print('TODO: Update use_dEdx for 2D and 3D')
                self.use_diffeq_dExdx = False
        elif disc_type.lower() == 'had' or disc_type.lower() == 'hadamard':
            assert self.settings['skew_sym'],"If hadamard scheme must also use skew-sym metrics. Set settings['skew_sym']=True"
            self.disc_type = 'had'  
            if hasattr(self.diffeq, had_flux.lower()+"_flux") or hasattr(self.diffeq, had_flux.lower()+"_fluxes"):
                self.had_flux = had_flux.lower()
            else:
                print("WARNING: 2-point flux '"+had_flux+"' not available for this Diffeq. Reverting to Central flux.")
                self.had_flux = 'central'
        
            if self.dim == 1:
                self.calc_had_flux = getattr(self.diffeq, self.had_flux + '_flux')
            if self.dim > 1:
                self.calc_had_flux = getattr(self.diffeq, self.had_flux + '_fluxes')
            # quick test
            '''
            try:
                if self.neq_node == 1:
                    from Source.Methods.Functions import build_F_sca, build_F_sca_2d
                    if self.dim == 1:
                        test = build_F_sca(np.ones((2,3)), np.ones((2,3)), self.calc_had_flux)
                    elif self.dim == 2:
                        test1, test2 = build_F_sca_2d(np.ones((2,3)), np.ones((2,3)), self.calc_had_flux)
                else:
                    from Source.Methods.Functions import build_F_sys, build_F_sys_2d
                    if self.dim == 1:
                        test = build_F_sys(self.neq_node, np.ones((2*self.neq_node,3)), np.ones((2*self.neq_node,3)), self.calc_had_flux)
                    elif self.dim == 2:
                        test1, test2 = build_F_sys_2d(self.neq_node, np.ones((2*self.neq_node,3)), np.ones((2*self.neq_node,3)), self.calc_had_flux)
            except:
                raise Exception('The Hadamard Flux did not compile properly.')
            '''
            
        else: raise Exception('Discretization type not understood. Try div or had.')
        if surf_diss == None or surf_diss == 'ND' or surf_diss == 'nd' or surf_diss == 'central':
            self.surf_diss = {'diss_type':'nd'}
        elif surf_diss == 'lf':
            self.surf_diss = {'diss_type':'lf'}
        elif surf_diss == 'upwind':
            self.surf_diss = {'diss_type':'upwind'}
        else:
            self.surf_diss = surf_diss
            assert(isinstance(self.surf_diss, dict)),"surf_diss must be a dictionary"
            assert(isinstance(self.surf_diss['diss_type'], str)),"surf_diss must contain a key 'diss_type'"
        if vol_diss == None or vol_diss == 'ND' or vol_diss == 'nd':
            self.vol_diss = {'diss_type':'ND'}
        else:
            self.vol_diss = vol_diss
            assert(isinstance(self.vol_diss, dict)),"vol_diss must be a dictionary"
            assert(isinstance(self.vol_diss['diss_type'], str)),"vol_diss must contain a key 'diss_type'"
        # use defaults in ADiss module.
        #self.settings.setdefault('jac_type',None) # what type? scalar, scalar-scalar, scalar-matrix, matrix-matrix
        #self.settings.setdefault('s',None) # usually set to p+1
        #self.settings.setdefault('coeff',0.1) # coefficient out infront
        #self.settings.setdefault('fluxvec',None) # what type of flux differencing? lf? only used if diss_type='upwind'
        self.pde_order1 = self.diffeq.pde_order1
        self.pde_order2 = self.diffeq.pde_order2


        assert isinstance(p, int), 'p must be an integer'
        self.p = p
        if nen == None:
            self.nen = 0
        else:
            assert isinstance(nen, int), f'nen must be an integer, {nen}'
            self.nen = nen
        
        if self.dim == 1:
            assert isinstance(nelem, int), 'nelem must be an integer'
            assert isinstance(xmin, float) or isinstance(xmin, int), 'xmin must be a float or integer'
            assert isinstance(xmax, float) or isinstance(xmax, int), 'xmax must be a float or integer'
            self.nelem = nelem
            self.xmin = xmin
            self.xmax = xmax

        elif self.dim == 2:
            if isinstance(nelem, int):
                self.nelem = (nelem, nelem)
            elif (isinstance(nelem, tuple) and len(nelem)==2):
                self.nelem = nelem
            else: 
                raise Exception('nelem must be an integer or tuple of integers. Inputted: ', nelem)
            if isinstance(xmin, float) or isinstance(xmin, int):
                self.xmin = (xmin, xmin)
            elif (isinstance(xmin, tuple) and len(xmin)==2):
                self.xmin = xmin
            else: 
                'xmin must be a float/integer or tuple of float/integers. Inputted: ', xmin
            if isinstance(xmax, float) or isinstance(xmax, int):
                self.xmax = (xmax, xmax)
            elif (isinstance(xmax, tuple) and len(xmax)==2):
                self.xmax = xmax
            else: 
                'xmax must be a float/integer or tuple of float/integers. Inputted: ', xmax
                
        elif self.dim == 3:
            if isinstance(nelem, int):
                self.nelem = (nelem, nelem, nelem)
            elif (isinstance(nelem, tuple) and len(nelem)==3):
                self.nelem = nelem
            else: 
                'nelem must be an integer or tuple of integers. Inputted: ', nelem
            if isinstance(xmin, float) or isinstance(xmin, int):
                self.xmin = (xmin, xmin, xmin)
            elif (isinstance(xmin, tuple) and len(xmin)==3):
                self.xmin = xmin
            else: 
                'xmin must be a float/integer or tuple of float/integers. Inputted: ', xmin
            if isinstance(xmax, float) or isinstance(xmax, int):
                self.xmax = (xmax, xmax, xmax)
            elif (isinstance(xmax, tuple) and len(xmax)==3):
                self.xmax = xmax
            else: 
                'xmax must be a float/integer or tuple of float/integers. Inputted: ', xmax
        else: raise Exception('Only set up currently for 1D, 2D, and 3D')

        # Other
        self.cons_obj_name = cons_obj_name
        self.bool_plot_sol = bool_plot_sol
        self.print_sol_norm = print_sol_norm

        ''' Extract other attributes '''

        if self.cons_obj_name is None:
            self.n_cons_obj = 0
            self.cons_obj_name = None
            self.bool_calc_cons_obj = False
        else:
            self.bool_calc_cons_obj = True
            # Standardize format for the name of the objective(s)
            if isinstance(self.cons_obj_name, str):
                self.n_cons_obj = 1
                self.cons_obj_name = (self.cons_obj_name.lower(),)
            elif isinstance(cons_obj_name, tuple):
                self.n_cons_obj = len(self.cons_obj_name)
            else:
                print(cons_obj_name)
                raise Exception('The variable cons_obj_name has to be either a string or a tuple of strings')
        
        if (self.settings['had_alpha'] != 1 or self.settings['had_beta'] != 1 or self.settings['had_gamma'] != 1):
            # if we use a generalized Hadamard form, we need to keep the exact metrics
            self.settings['calc_exact_metrics'] = True

        self.diffeq.calc_cons_obj = self.calc_cons_obj
        self.diffeq.n_cons_obj = self.n_cons_obj
        self.diffeq.cons_obj_name = self.cons_obj_name

        self.check_resid_conv = check_resid_conv
        self.print_residual = print_residual

        ''' Check all inputs '''

        # Domain
        if bc == None:
            print('No boundary conditions provided. Checking diffeq for a default.')
            self.bc = diffeq.bc
        else: self.bc = bc
        if self.dim == 1:
            assert isinstance(self.bc, str), 'bc must be a string'
            if self.bc == 'periodic': 
                self.periodic = True
            else:
                self.periodic = False
        elif self.dim == 2:
            if isinstance(self.bc, str):
                self.bc = (self.bc, self.bc)
            assert isinstance(self.bc, tuple) and len(self.bc)==2, 'bc must be a string or tuple of strings'
            self.periodic = [False,False]
            if self.bc[0] == 'periodic': self.periodic[0] = True
            if self.bc[1] == 'periodic': self.periodic[1] = True               
        elif self.dim == 3:
            if isinstance(self.bc, str):
                self.bc = (self.bc, self.bc, self.bc)
            assert isinstance(self.bc, tuple) and len(self.bc)==3, 'bc must be a string or tuple of strings'
            self.periodic = [False,False,False]
            if self.bc[0] == 'periodic': self.periodic[0] = True
            if self.bc[1] == 'periodic': self.periodic[1] = True
            if self.bc[2] == 'periodic': self.periodic[2] = True
            
        
        ''' Call the discretisation specific method to finish the class initialization '''
        self.init_disc_specific()  
        if self.settings['stop_after_metrics']:
            return
        
        # Time marching data
        if self.t_final == None:
            print('No t_final given. Checking diffeq for a default t_final.')
            try:
                self.t_final = self.diffeq.t_final
            except:
                print('WARNING: No default t_final found in diffeq. Should set manually before calling solve().')
        if (isinstance(self.t_final, int) or isinstance(self.t_final, float)) and hasattr(self.diffeq, 't_final'):
            if (self.t_final != self.diffeq.t_final) and (self.diffeq.t_final is not None):
                print('WARNING: The diffeq default is t_final =',self.diffeq.t_final,', but you selected t_final =',t_final)
        self.set_timestep(dt)
                
        # Free Stream Preservation sanity check
        if self.bc == 'periodic':
            test = self.free_stream(print_result=False)
            if test>1e-12:
                print('WARNING: Free Stream is not preserved. Check Metrics and/or SAT discretization.')
                print('         Free Stream is violated by a maximum of {0:.2g}'.format(np.max(abs(test))))

        if self.print_progress: print('---------- Solver succesfully initialized. ----------')

    def solve(self, q0=None):

        ''' Solve to calculate the solution and the objective '''

        #start_time = time.time()
        
        if q0 is None:
            q0 = self.diffeq.set_q0()

        if self.dt_to_be_set:
            raise Exception('Time step not set yet. Use set_timestep(dt) before running solve().')
        
        if self.t_initial != 0.0:
            self.set_timestep(self.dt)
        
        tm_class = TimeMarching(self.diffeq, self.tm_method, self.keep_all_ts,
                        skip_ts = self.skip_ts,
                        bool_plot_sol = self.bool_plot_sol,
                        bool_calc_cons_obj = self.bool_calc_cons_obj,
                        print_sol_norm = self.print_sol_norm,
                        print_residual = self.print_residual,
                        check_resid_conv = self.check_resid_conv,
                        dqdt=self.dqdt, dfdq=self.dfdq,
                        rtol=self.tm_rtol, atol=self.tm_atol)
        
        tm_class.nframes = self.tm_nframes
        tm_class.print_progress = self.print_progress
        self.q_sol =  tm_class.solve(q0, self.dt, self.n_ts, self.t_initial)
        self.cons_obj = tm_class.cons_obj
        self.t_final = tm_class.t_final

        #end_time = time.time()
        #self.simulation_time = end_time - start_time
        #print(f"solver.solve() complete. Took {self.simulation_time} seconds.")
    
    def calc_cons_obj(self, q, t, dqdt, cons_obj_name=None):
        '''
        Purpose
        ----------
        Calculate the conservation objectives, such as energy, conservation,
        entropy, etc. This calls on smaller specific functions defined for
        each spatial discretization

        Parameters
        ----------
        q : np array
            The global solution at the global nodes.

        Returns
        -------
        A vector of size n_cons_obj with the desired conservation objectives
        '''

        if q is None:
            q = self.q_sol
        
        if cons_obj_name is None:
            cons_obj_name = self.cons_obj_name
            n_cons_obj = self.n_cons_obj
        else:
            n_cons_obj = len(cons_obj_name)
        
        if q.ndim == 2:
            cons_obj = np.zeros(n_cons_obj)
            if any('max_eig'==name.lower() or 'spec_rad'==name.lower() for name in cons_obj_name):
                eigs = self.check_eigs(q, plot_eigs=False, returneigs=True, print_nothing=True)
        elif q.ndim == 3:
            cons_obj = np.zeros((n_cons_obj,q.shape[2]))
            if any('der' in name.lower() for name in self.cons_obj_name):
                raise Exception('ERROR: Cannot calculate conservation objectives with derivatives since dqdt was not stored.')
            if any('max_eig'==name.lower() or 'spec_rad'==name.lower() for name in cons_obj_name):
                raise Exception('TODO: Need to loop over time steps to calculate eigenvalues')
            
        if any('zelalem_coeff' in name.lower() for name in self.cons_obj_name):
            zel_coeff = self.zelalem_diss_coeff(q)

        for i in range(n_cons_obj):
            cons_obj_name_i = cons_obj_name[i].lower()

            if cons_obj_name_i == 'energy':
                cons_obj[i] = self.energy(q)
            elif cons_obj_name_i == 'entropy':
                cons_obj[i] = self.entropy(q)
            elif cons_obj_name_i == 'conservation':
                cons_obj[i] = self.conservation(q)
            elif cons_obj_name_i == 'energy_der':
                cons_obj[i] = self.energy_der(q,dqdt)
            elif cons_obj_name_i == 'entropy_der':
                cons_obj[i] = self.entropy_der(q,dqdt)
            elif cons_obj_name_i == 'conservation_der':
                cons_obj[i] = self.conservation_der(dqdt)
            elif cons_obj_name_i == 'kinetic_energy':
                assert ('Euler' in self.diffeq.diffeq_name), 'Kinetic Energy only defined for Euler Equations'
                cons_obj[i] = self.kinetic_energy(q)
            elif cons_obj_name_i == 'enstrophy':
                assert ('Euler' in self.diffeq.diffeq_name), 'Enstrophy only defined for Euler Equations'
                cons_obj[i] = self.enstrophy(q)
            elif cons_obj_name_i == 'a_energy':
                assert (self.diffeq.diffeq_name == 'VariableCoefficientLinearConvection' or self.diffeq.diffeq_name == 'Burgers'), 'A_Energy only defined for Variable Coefficient Problems'
                cons_obj[i] = self.diffeq.a_energy(q)
            elif cons_obj_name_i == 'a_energy_der':
                assert (self.diffeq.diffeq_name == 'VariableCoefficientLinearConvection' or self.diffeq.diffeq_name == 'Burgers'), 'A_Energy only defined for Variable Coefficient Problems'
                cons_obj[i] = self.diffeq.a_energy_der(q,dqdt)
            elif cons_obj_name_i == 'a_conservation':
                assert (self.diffeq.diffeq_name == 'VariableCoefficientLinearConvection' or self.diffeq.diffeq_name == 'Burgers'), 'A_Conservation only defined for Variable Coefficient Problems'
                cons_obj[i] = self.diffeq.a_conservation(q,dqdt)
            elif cons_obj_name_i == 'max_eig':
                cons_obj[i] = np.max(eigs.real)
            elif cons_obj_name_i == 'spec_rad':
                cons_obj[i] = np.max(np.abs(eigs))
            elif cons_obj_name_i == 'sol_error' or cons_obj_name_i == 'error':
                cons_obj[i] = self.calc_error(q,t)
            elif cons_obj_name_i == 'time':
                cons_obj[i] = t
            elif 'zelalem_coeff' in cons_obj_name_i:
                if 'max' in cons_obj_name_i:
                    cons_obj[i] = np.max(zel_coeff)
                elif 'min' in cons_obj_name_i:
                    cons_obj[i] = np.min(zel_coeff)
                elif 'mean' in cons_obj_name_i:
                    cons_obj[i] = np.mean(zel_coeff)
            else:
                raise Exception('Unknown conservation objective function')

        return cons_obj
    
    
    def norm(self, q, nen=None):
        ''' calculate the energy norm q.T @ H @ q given a global q'''
        return np.sqrt(float(self.energy(q,nen)))
    
    def calc_error(self, q=None, tf=None, method=None, use_all_t=False,
                   var2plot_name=None):
        '''
        Purpose
        ----------
        Calculate the error of the solution using a defined method. This calls
        on smaller specific functions also defined below.

        Parameters
        ---------
        q : np array (optional)
            The global solution at the global nodes. If None, this uses the
            final solution determined by solve()
        tf : float (optional)
            The time at which to evaluate the error. If None, this uses the
            default final time of solve()
        method : string (optional)
            Determines which error to use. Options are:
                'SBP' : the SBP error sqrt((q-q_ex).T @ H @ (q-q_ex))
                'Rms' : the standard root mean square error in L2 norm
                'Boundary' : The simple periodic boundary error | q_1 - q_N |
                'Truncation-SBP' : the SBP error but instead using er = dqdt(q-q_ex)
                'Truncation-Rms' : the Rms error but instead using er = dqdt(q-q_ex)
                'max_diff' : np.max(abs(q-q_ex))
        use_all_t : bool (optional)
            If True, it calls calc_error recursively for all time steps 
            (len of q_sol) and stores the results in a 1D array.
            The default is False.
        '''
        if self.diffeq.has_exa_sol == False: return None
        if tf == None: tf = self.t_final
        if q is None:
            if self.q_sol.ndim == 2: q = self.q_sol
            elif self.q_sol.ndim == 3: q = self.q_sol[:,:,-1]
        if method == None:
            if self.disc_type == 'FD': method = 'Rms'
            else: method = 'SBP'
        if use_all_t:
            assert(self.q_sol.ndim == 3),'ERROR: There is only one time step for q_sol.'
            steps = np.shape(self.q_sol)[2]
            errors = np.zeros(steps)
            times = np.linspace(0,self.t_final,steps)
            # TODO: generalize this for varying time steps, if we ever implement this
            for i in range(steps):
                errors[i] = self.calc_error(self.q_sol[:,:,i],times[i],method=method)
            return errors
        
        q_exa = self.diffeq.exact_sol(tf,guess=q)
        if var2plot_name is None:
            var = q
            var_exa = q_exa
            neq = self.neq_node
        else:
            var = self.diffeq.var2plot(q,var2plot_name)
            var_exa = self.diffeq.var2plot(q_exa,var2plot_name)
            if var2plot_name == 'q':
                neq = self.neq_node
            else:
                neq = 1

        # determine error to use
        if method == 'SBP' or method == 'Rms' or method == 'L2':
            error = var - var_exa
        elif method == 'max diff':
            error = np.max(abs(var-var_exa))
        elif method == 'Boundary':
            error = abs(var[0]-var[-1])
        elif method == 'Truncation-SBP' or method == 'Truncation-Rms':
            # I think this is wrong... not typical truncation error
            error = self.diffeq.dqdt(var - var_exa)
        else:
            raise Exception('Unknown error method. Use one of: H-norm, Rms, Boundary, Truncation-SBP, Truncation-Rms')

        # if we still need to apply a norm, do it
        if method == 'SBP' or method == 'Truncation-SBP':
            error = self.norm(error,neq)
        elif method == 'Rms' or method == 'Truncation-Rms' or method == 'L2':
            error = np.linalg.norm(error) / np.sqrt(self.nn)
        return error

    def reset(self,variables=[],copy_diffeq=False):
        """
        Purpose
        ----------
        Allows one to reset the solver instance. It will read the parameters
        in variables, then call __init__ with the updated arguments.

        Parameters
        ----------
        variables : list of tuples
        The attributes to be updated for the solver. Default is empty.
        tuples must be in form ('name_of_attribute',value)
        """
        for i in range(len(variables)):
            attribute , value = variables[i]
            if hasattr(self, attribute):
                setattr(self, attribute, value)
            else:
                print("ERROR: solver has no attribute '{0}'. Ignoring.".format(attribute))
        
        if copy_diffeq:
            from copy import copy
            new_diffeq = copy(self.diffeq)
            # Note: this is a shallow copy, i.e. the references point to the same original objects
            # HOWEVER, if you overwrite attributes, you will not change the original objects. 
            # Instead, you will create new objects and change the references, as we want.
        else:
            new_diffeq = self.diffeq
        
        self.__init__(new_diffeq, self.settings, 
                      self.tm_method, self.dt, self.t_final, 
                      p=self.p, disc_type=self.disc_type, had_flux=self.had_flux,
                      surf_diss=self.surf_diss, vol_diss=self.vol_diss,
                      nelem=self.nelem, nen=self.nen, disc_nodes=self.disc_nodes,
                      bc=self.bc, xmin=self.xmin, xmax=self.xmax,
                      cons_obj_name=self.cons_obj_name,
                      bool_plot_sol=self.bool_plot_sol, print_sol_norm=self.print_sol_norm,
                      print_residual=self.print_residual, check_resid_conv=self.check_resid_conv,
                      sparse=self.sparse, sat_sparse=self.sat_sparse)
        
    def set_timestep(self, dt=None):
        """
        Purpose
        ----------
        Sets the appropriate time step, and runs all the necessary checks.
        Allows one to easily reset the time step for a new solver instance.

        Parameters
        ----------
        dt : float
            The new time step to be used.
        """

        if dt == None and self.dt == None:
            print('WARNING: No time step given. Continuing with initialization, ignoring time marching.')
            print('         Use set_timestep(dt) to set a time step before running solve().')
            self.dt = None
            self.n_ts = None
            self.dt_to_be_set = True
            return
        
        elif dt is not None:
            self.dt = float(dt)

        if self.t_final == None:
            print('WARNING: No t_final given. Continuing with initialization, ignoring time marching.')
            print('         Use set_timestep() to set a time step before running solve().')
            self.n_ts = None
            self.dt_to_be_set = True
            return
        
        self.dt_to_be_set = False

        # set the number of time steps
        if isinstance(self.t_final, int) or isinstance(self.t_final, float):
            self.n_ts = int(np.round((self.t_final-self.t_initial) / self.dt))
            if abs(self.n_ts - ((self.t_final-self.t_initial) / self.dt)) > 1e-12:
                dt_old = np.copy(self.dt)
                self.dt = (self.t_final-self.t_initial)/self.n_ts
                print('WARNING: To ensure final time is exact, changing dt from {0} to {1}'.format(dt_old,self.dt))
        elif self.t_final == 'steady':
            self.n_ts = 100000
            print('Indicated steady problem. Will use a convergence criteria to stop time march, or max {0} steps unless otherwise corrected.'.format(self.n_ts))
            self.check_resid_conv = True
        else:
            raise Exception('ERROR: t_final not understood,', self.t_final) 
        
        if self.diffeq.nondimensionalize:
            if (self.diffeq.t_scale != 1.0) and (self.t_final != 'steady'):
                print(f'WARNING: since nondimensionalizing, changing time from t_final and dt')
                print(f'         from t_final={self.t_final} and dt={self.dt} to t_final={self.t_final * self.diffeq.t_scale} and dt={self.dt * self.diffeq.t_scale}.')
                self.t_final = self.t_final * self.diffeq.t_scale
                self.t_initial = self.t_initial * self.diffeq.t_scale
                self.dt = self.dt * self.diffeq.t_scale

        if self.diffeq.steady and not self.check_resid_conv:
            print('WARNING: Doing a steady solve with no residual checking. Consider setting check_resid_conv=True.')
        
        if not self.diffeq.steady and self.check_resid_conv:
            print('WARNING: set a convergence criteria for an unsteady solve. Manually fixing to check_resid_conv=False.')
            self.check_resid_conv = False

        # time step stability sanity check
        q = self.diffeq.set_q0()
        cfl = 0.5
        if self.dim==1:
            LFconst = np.max(self.diffeq.maxeig_dExdq(q))
            dt = cfl*(self.xmax-self.xmin)/self.nn/LFconst
            if self.dt > dt:
                print('WARNING: time step dt={0:.2g} may not be small enough to remain stable.'.format(self.dt))
                print('Assuming CFL = {2:g} and max wave speed = {0:.2g}, try dt < {1:.2g}'.format(LFconst, dt, cfl))
        elif self.dim==2:
            LFconstx = np.max(self.diffeq.maxeig_dExdq(q))
            LFconsty = np.max(self.diffeq.maxeig_dEydq(q))
            const = np.sqrt(LFconstx**2 + LFconsty**2)
            dx = min((self.xmax[0]-self.xmin[0])/self.nn[0],(self.xmax[1]-self.xmin[1])/self.nn[1])
            dt = cfl*dx/const
            if self.dt > dt:
                print('WARNING: time step dt={0:.2g} may not be small enough to remain stable.'.format(self.dt))
                print('Assuming CFL = {3:g} and max wave speed = ({0:.2g}, {1:.2g}), try dt < {2:.2g}'.format(LFconstx, LFconsty, dt, cfl))
        elif self.dim==3:
            LFconstx = np.max(self.diffeq.maxeig_dExdq(q))
            LFconsty = np.max(self.diffeq.maxeig_dEydq(q))
            LFconstz = np.max(self.diffeq.maxeig_dEzdq(q))
            const = np.sqrt(LFconstx**2 + LFconsty**2 + LFconstz**2)
            dx = min((self.xmax[0]-self.xmin[0])/self.nn[0],(self.xmax[1]-self.xmin[1])/self.nn[1])
            dt = cfl*dx/const
            if self.dt > dt:
                print('WARNING: time step dt={0:.2g} may not be small enough to remain stable.'.format(self.dt))
                print('Assuming CFL = {2:g} and max wave speed = ({0:.2g}, {1:.2g}, {2:.2g}), try dt < {3:.2g}'.format(LFconstx, LFconsty, LFconstz, dt, cfl))


        
        
    def calc_LHS(self, q=None, t=0., exact_dfdq=True, step=1.0e-4, istep=1.0e-15, 
                 finite_diff=False, print_nothing=False, print_error=False):
        '''
        Either get the exact LHS operator on q if the problem is linear, or the
        linearization (LHS) of it at a particular state q. Either done exactly with
        self.dfdq(q) or approximately with finite differences, calling self.diffeq.dqdt 
        A_ij \approx ( dqdt(q + e_j*tol) - dqdt(q - e_j*tol) ) / 2*tol

        Parameters
        ----------
        q : numpy array, optional
            Current state (must be compatible with self.diffeq.dqdt(q))
            Default is last solution.
        exact_dfdq : bool, optional
            Get the LHS analytically or approximately. The default is True.
        step : float, optional
            the step size for the finite difference approx. The default is 1.0e-4.

        Returns
        -------
        A : 2D numpy array
            Approximate RHS spatial operator
        '''
        if q is None:
            if hasattr(self, 'q_sol'):
                if self.q_sol is not None:
                    if self.q_sol.ndim == 2: q = self.q_sol
                    elif self.q_sol.ndim == 3: q = self.q_sol[:,:,-1]
                else:
                    q = self.diffeq.set_q0()
            else:
                q = self.diffeq.set_q0()
        
        if exact_dfdq:
            try:  
                A = self.dfdq(q)
            except:
                exact_dfdq = False
                if finite_diff:
                    if not print_nothing: ('WARNING: self.dfdq(q) returned errors. Using finite difference.')
                else:
                    if not print_nothing: print('WARNING: self.dfdq(q) returned errors. Using complex step.')
        if not exact_dfdq:
            from Source.Methods.Analysis import printProgressBar
            nen,nelem = q.shape   
            nn = nelem*nen   
            assert(self.qshape==q.shape),"ERROR: sizes don't match"
            A = np.zeros((nn,nn)) 
            if not finite_diff:
                try:  
                    for i in range(nen):
                        if not print_nothing: 
                            if nn>=400:
                                printProgressBar(i, nen-1, prefix = 'Complex Step Progress:')
                        for j in range(nelem):
                            ei = np.zeros((nen,nelem),dtype=np.complex128)
                            ei[i,j] = istep*1j
                            qi = self.dqdt(np.complex128(q)+ei, t).flatten('F')
                            idx = np.where(np.imag(ei.flatten('F'))>istep/10)[0][0]
                            A[:,idx] = np.imag(qi)/istep
                except Exception as e:  
                    if not print_nothing: print('WARNING: complex step returned errors. Using finite difference.') 
                    if print_error: print(e)
                    finite_diff = True
            if finite_diff:        
                for i in range(nen):
                    if not print_nothing: 
                        if nn>=400:
                            printProgressBar(i, nen-1, prefix = 'Complex Step Progress:')
                    for j in range(nelem):
                        ei = np.zeros((nen,nelem))
                        ei[i,j] = 1.*step
                        q_r = self.dqdt(q+ei, t).flatten('F')
                        q_l = self.dqdt(q-ei, t).flatten('F')
                        idx = np.where(ei.flatten('F')>step/10)[0][0]
                        A[:,idx] = (q_r - q_l)/(2*step)
        return A

    
    def check_eigs(self, q=None, plot_eigs=True, returnA=False, returneigs=False, plot_maxvec=False, 
                   exact_dfdq=False, finite_diff=False, step=5.0e-6, istep=1e-15, tol=1.0e-10, 
                   savefile=None, print_nothing=False, colour_by_k=False, normalize=False,
                   ymin=None, ymax=None, xmin=None, xmax=None, print_error=False,
                   time=None, display_time=False, display_maxreal=False,
                   title=None, save_format='png', dpi=600, overwrite=False, **kargs):
        '''
        Call on self.diffeq.dqdt to check the stability of the spatial operator
        at a particular state q using central finite differences (approximate!).
        A_ij \approx ( dqdt(q + e_j*tol) - dqdt(q - e_j*tol) ) / 2*tol

        Parameters
        ----------
        q : numpy array, optional
            Current state (must be compatible with self.diffeq.dqdt(q))
            Default is last solution.
        plot_eigs : boolean, optional
            Flag whether to plot eigenvalues. The default is True.
        returnA : boolean, optional
            Flag whether to return the spatial operator. The default is False.
        step : float, optional
            step size for finite difference approximation. The default is 1.0e-2.
        tol : float, optional
            The tolerance allowed for detecting postive eigenvaules. 
            The default is 1.0e-10.
        plt_save_name : str, optional
            name of saved file, without file extension
            The defualt is None. 
        ymin, ymax, xmin, xmax : str, optional
            Y and X axis limits.
            The defualt is None.
        **kargs : nothing
            This just absorbs additional unecessary keword arguments

        Returns
        -------
        A : 2D numpy array
            Approximate RHS spatial operator

        '''
        if not print_nothing: print('Checking Eigenvalues of System LHS Operator')
        A = self.calc_LHS(q=q, exact_dfdq=exact_dfdq, step=step, istep=istep, 
                          finite_diff=finite_diff, print_nothing=print_nothing,
                          print_error=print_error)
        if normalize:
            if self.dim==1:
                ds = self.neq_node*self.nelem*(self.nen-1)/(self.xmax-self.xmin)
            elif self.dim==2:
                dx = self.nelem[0]*(self.nen-1)/(self.xmax[0]-self.xmin[0])
                dy = self.nelem[1]*(self.nen-1)/(self.xmax[1]-self.xmin[1])
                ds = self.neq_node*np.sqrt(dx*dx+dy*dy)
            A /= ds
        nen1, nen2 = A.shape
        if not print_nothing: 
            if nen1 >= 5000:
                import datetime
                time_est = 1.02e-7*nen1**2.24
                start_time = datetime.datetime.now()
                end_time = start_time + datetime.timedelta(seconds=time_est)
                print('HEADS UP: Large matrix size. This may take a while.')
                print(f"Estimating eigenvalue solve will take {int(time_est)} seconds, i.e. finish around {end_time.strftime('%H:%M:%S')}" )
        if ((colour_by_k) and self.dim==1 and self.neq_node==1 and plot_eigs) or plot_maxvec:
            eigs, eigvecs = np.linalg.eig(A)
        else:
            eigs = np.linalg.eigvals(A)
        if not print_nothing: 
            print('Max real component =',np.max(eigs.real))
            if max(eigs.real) < tol:
                print('Max eigenvalue within tolerance. STABLE.')
            else:
                print('Max eigenvalue exceeds tolerance. UNSTABLE.')

            spec_rad = np.max(np.abs(eigs))
            dt_max = 2.5/spec_rad
            print(u"Assuming RK4, to remain stable (\u03bbh \u2264 2.5), should use dt \u2264", round(dt_max, -int(np.floor(np.log10(abs(dt_max/10))))))
            
        if plot_eigs:
            if colour_by_k:
                second_moment = True # use the first moment (like COM) or second moment (like variance from 0)
                from scipy.stats import moment
                if self.dim >1 or self.neq_node>1: 
                    if not print_nothing: print('WARNING: colour by wavenumber not set up for dim >1 or neq>0. Ignoring.')
                    avg_k=None
                else:
                    def remove_duplicates_and_average(x, y):
                        # important for SBP to avoid doubled nodes
                        duplicates = np.where(np.diff(x) == 0)[0] # Find indices where x has repeated values
                        mask = np.ones(len(x), dtype=bool) # Initialize a mask to identify unique elements
                        # For each duplicate, average the y values and mark the second occurrence for removal
                        for idx in duplicates:
                            y[idx] = (y[idx] + y[idx + 1]) / 2
                            mask[idx + 1] = False
                        new_x = x[mask] # Apply mask to remove duplicates
                        new_y = y[mask]
                        return new_x, new_y
                    if self.disc_nodes == 'lgl':
                        # evaluate eigenvectors at equispaced nc nodes so fft works
                        nc_nodes = np.linspace(0,1,self.nen,endpoint=True)
                        elem_nodes = np.reshape(self.sbp.x,(self.nen,1))
                        wBary = MakeDgOp.BaryWeights(elem_nodes)
                        V_to_nc = MakeDgOp.VandermondeLagrange1D(nc_nodes,elem_nodes,wBary)
                        for elem in range(self.nelem):
                            eigvecs[elem*self.nen:(elem+1)*self.nen] = V_to_nc @ eigvecs[elem*self.nen:(elem+1)*self.nen]
                    elif self.disc_nodes == 'lg':
                        # evaluate eigenvectors at equispaced nodes (not nc) so fft works
                        nc_nodes = np.linspace(0,1,self.nen,endpoint=False)
                        elem_nodes = np.reshape(self.sbp.x,(self.nen,1))
                        wBary = MakeDgOp.BaryWeights(elem_nodes)
                        V_to_nc = MakeDgOp.VandermondeLagrange1D(nc_nodes,elem_nodes,wBary)
                        for elem in range(self.nelem):
                            eigvecs[elem*self.nen:(elem+1)*self.nen] = V_to_nc @ eigvecs[elem*self.nen:(elem+1)*self.nen]

                    
                    new_x, new_eigvecs = remove_duplicates_and_average(self.mesh.x, eigvecs)
                    n = len(new_x)
                    avg_k = np.zeros(self.nn)
                    for i in range(self.nn):
                        power_spec = np.abs(np.fft.fft(new_eigvecs[:,i])[1:])**2 # ignore the zero frequency, but NOT negative frequencies (important when eigenvectors are complex)
                        if n % 2 == 0: # even
                            ks = np.linspace(1,int(n/2),int(n/2),endpoint=True)
                            ks = np.concatenate((ks, ks[-2::-1]))
                        else: # odd
                            ks = np.linspace(1,int((n-1)/2),int((n-1)/2),endpoint=True)
                            ks = np.concatenate((ks, ks[-1::-1]))
                        if np.sum(power_spec < 1e-10):
                            avg_k[i] = 0
                        else:
                            if second_moment:
                                avg_k[i] = np.sqrt(np.sum(power_spec*ks*ks)/np.sum(power_spec))
                                # square root to rescale to be between min(ks) and max(ks)
                            else:
                                avg_k[i] = np.average(ks,weights=power_spec)
                                # np.sum(power_spec*ks)/np.sum(power_spec) # equivalent to above

            else:
                avg_k=None

            plt.figure()
            X = [x.real for x in eigs]
            Y = [x.imag for x in eigs]
            if avg_k is None:
                plt.scatter(X,Y, color='red')
            else:
                plt.scatter(X,Y, c=avg_k, cmap='viridis', vmin=1, vmax=np.max(ks))
                plt.colorbar(label='Average Wavenumber')
            plt.axvline(x=0, linewidth=1, linestyle='--', color='black')
            plt.xlabel(r'Real Component ($x<0$ for stability)',fontsize=14)
            plt.ylabel(r'Imaginary Component',fontsize=14)
            if title is None:
                plt.title(r'Eigenvalues',fontsize=16)
            else:
                plt.title(title,fontsize=16)
            plt.ylim(ymin,ymax)
            plt.xlim(xmin,xmax)
            if display_time and (time is not None):
                ax = plt.gca()
                # define matplotlib.patch.Patch properties
                #props = dict(boxstyle='round', facecolor='white')
                ax.text(0.03, 0.95, f'$t={round(time,2)}$', transform=ax.transAxes, 
                        fontsize=14, verticalalignment='top') 
            if display_maxreal:
                ax = plt.gca()
                ax.text(0.03, 0.88, f'$\\max \\Re(\\lambda) = {np.max(eigs.real):.2g}$', transform=ax.transAxes, 
                        fontsize=14, verticalalignment='top') 

            if 'scinet' not in gethostname():
                plt.tight_layout()   # for some reason doesn't work on cluster     
            if savefile is not None:
                filename = savefile+'.'+save_format
                if path.exists(filename) and not overwrite:
                    print('WARNING: File name already exists. Using a temporary name instead.')
                    plt.savefig(filename+'_RENAMEME', format=save_format, dpi=dpi)
                else: 
                    plt.savefig(filename, format=save_format, dpi=dpi)
            plt.show()

        if plot_maxvec:
            #TODO: assumes 1D, also not well suited to neq_node>1
            num_vecs = 5
            # Find the indices of the {num_vecs} largest eigenvalues
            largest_indices = np.argsort(eigs)[-num_vecs:][::-1]
            for i, idx in enumerate(largest_indices):
                # Extract the corresponding eigenvectors
                eigenvalue = eigs[idx]
                eigenvector = eigvecs[:, idx]
                
                # Calculate real part, imaginary part, and power spectrum
                real_part = np.real(eigenvector)
                imaginary_part = np.imag(eigenvector)
                amplitude = np.abs(eigenvector)
                
                # Create the plot
                plt.figure()
                plt.plot(self.mesh.x, real_part, label='Real Component')
                plt.plot(self.mesh.x, imaginary_part, label='Imaginary Component')
                plt.plot(self.mesh.x, amplitude, label='Amplitude', linestyle='--')
                
                # Add title, legend, and labels
                plt.title(f'Eigenvalue: {eigenvalue.real:.2f} + {eigenvalue.imag:.2f}i')
                plt.xlabel(r'$x$')
                plt.ylabel('Value')
                plt.legend()
                
                # Show the plot for each eigenvalue
                plt.show()

        
        if returnA and returneigs:
            return A, eigs
        elif returnA:
            return A
        elif returneigs:
            return eigs
        
    # Define a function to compute eigenvalues and eigenvectors, and plot the results
    def plot_eigvecs(self, matrix=None, plot_type="real", 
                     plot_positive_eigs=True, plot_maxabs_eigs=False,
                     threshold=1e-5, num_eigvecs = 5,
                     include_pairs=True, return_eigs=False):
        assert self.dim==1, 'ERROR: Only implemented for 1D problems'
        assert (plot_positive_eigs and not plot_maxabs_eigs) or  \
                (not plot_positive_eigs and plot_maxabs_eigs), 'ERROR: Choose one of plot_positive_eigs or plot_maxabs_eigs'      
        
        if matrix is None:
            matrix = self.calc_LHS()

        if return_eigs:
            eigvector_list = []
            
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(matrix)

        if plot_positive_eigs:
            # Organize eigenvalues and eigenvectors by descending real part of eigenvalues
            sorted_indices = np.argsort(-np.real(eigenvalues))
        else:
            # Organize eigenvalues and eigenvectors by descending magnitude of eigenvalues
            sorted_indices = np.argsort(-np.abs(eigenvalues))
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # Track paired eigenvalues
        processed_indices = set()

        # Iterate over eigenvalues and eigenvectors
        for idx, eigenvalue in enumerate(eigenvalues):
            if idx in processed_indices:
                continue
            
            # Check for positive real part above threshold OR
            # maximum 5 eigenvalues by magnitude
            if (plot_positive_eigs and np.real(eigenvalue) > threshold) \
                or (plot_maxabs_eigs and idx < num_eigvecs):
                eigenvector = eigenvectors[:, idx]

                # Separate components of the eigenvector based on plot type
                vars = eigenvector.reshape((self.nelem*self.nen,self.neq_node), order='C')
                if plot_type == "real":
                    vars = vars.real
                    ylabel = "Real Part"
                elif plot_type == "imaginary":
                    vars = vars.imag
                    ylabel = "Imaginary Part"
                elif plot_type == "power":
                    vars = np.abs(vars)**2
                    ylabel = "Power Spectrum"
                else:
                    raise ValueError("Invalid plot_type. Choose 'real', 'imaginary', or 'power'.")

                for vari in range(self.neq_node):
                    if np.max(np.abs(vars[:,vari])) < 1e-12:
                        norm = 1.
                    else:
                        norm = np.linalg.norm(vars[:,vari])
                    vars[:,vari] = vars[:,vari] / norm
                x = self.mesh.x

                # Initialize figure
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
                plt.figure()
                for vari in range(self.neq_node):
                    plt.plot(x, vars[:,vari], label=f'Variable {vari+1}', linestyle='solid', color=colors[vari])
                
                if return_eigs:
                    eigvector_list.append(eigenvector)

                # Check for and include the complex pair if applicable
                if include_pairs and np.iscomplex(eigenvalue):
                    for jdx, other_eigenvalue in enumerate(eigenvalues):
                        if jdx != idx and np.isclose(eigenvalue.conj(), other_eigenvalue):
                            processed_indices.add(jdx)
                            paired_eigenvector = eigenvectors[:, jdx]

                            vars2 = paired_eigenvector.reshape((self.nelem*self.nen,self.neq_node), order='C')
                            if plot_type == "real":
                                vars2 = vars2.real
                            elif plot_type == "imaginary":
                                vars2 = vars2.imag
                            elif plot_type == "power":
                                vars2 = np.abs(vars2)**2

                            for vari in range(self.neq_node):
                                if np.max(np.abs(vars2[:,vari])) < 1e-12:
                                    norm = 1.
                                else:
                                    norm = np.linalg.norm(vars2[:,vari])
                                vars2[:,vari] = vars2[:,vari] / norm
                                plt.plot(x, vars2[:,vari], label=f'Variable {vari+1} (Pair)', linestyle='dotted', color=colors[vari])

                            break

                plt.title(f"Eigenvectors for Eigenvalue: {eigenvalue:g}")
                plt.xlabel("x")
                plt.ylabel(ylabel)
                plt.legend()
                plt.grid()
                plt.show()
        
        if return_eigs:
            return eigvector_list

    def plot_cons_obj(self,savefile=None,plot_change=True,final_idx=None,start_idx=0,
                      logscale=True,normalize_eigs=False):
        '''
        Plot the conservation objectives

        Parameters
        ----------
        savefile : string, optional
            File name under which to save plots. The default is None.
        '''
        if self.cons_obj is None:
            print('WARNING: No conservation objectives to plot.')
            return
        if normalize_eigs:
            assert self.dim==1, 'Normalizing only set up for 1D'
            eig_norm = (self.xmax-self.xmin) / (self.nelem*(self.nen-1))
        else:
            eig_norm = 1
        for i in range(self.n_cons_obj):
            cons_obj_name_i = self.cons_obj_name[i].lower()
            if cons_obj_name_i == 'time': continue
            norm = self.cons_obj[i,0]
            plt.figure(figsize=(6,4))
            plt.xlabel(r'Time $t$',fontsize=16)
            if 'time' in [name.lower() for name in self.cons_obj_name]:
                time = self.cons_obj[np.where('time' == np.array([name.lower() for name in self.cons_obj_name]))[0][0]]
            else:
                time = np.linspace(0,self.t_final,len(self.cons_obj[i]))

            if cons_obj_name_i == 'energy':
                if plot_change:
                    plt.title(r'Change in Energy',fontsize=18)
                    plt.ylabel(r'$\vert \vert u(x,t)^2 \vert \vert_H - \vert \vert u_0(x)^2 \vert \vert_H$',fontsize=16)
                    plt.plot(time[start_idx:final_idx],self.cons_obj[i,start_idx:final_idx]-norm) 
                    if logscale: 
                        plt.yscale('symlog',linthresh=1e-14)
                        ax = plt.gca()
                        ymin, ymax = ax.get_ylim()
                        positive_ticks = [0] + [10**exp for exp in range(-12, int(np.log10(ymax)) + 1, 4)]
                        negative_ticks = [-10**exp for exp in range(-12, int(np.log10(-ymin)) + 1, 4)]
                        custom_ticks = negative_ticks[::-1] + positive_ticks
                        ax.set_yticks(custom_ticks)
                else:
                    plt.title(r'Energy',fontsize=18)
                    plt.ylabel(r'$\vert \vert u(x,t)^2 \vert \vert_H $',fontsize=16)
                    plt.plot(time[start_idx:final_idx],self.cons_obj[i,start_idx:final_idx]) 
                    if logscale: plt.yscale('log')
                    #plt.gca().invert_yaxis()
                
            elif cons_obj_name_i == 'entropy':
                plt.title(r'Change in Entropy',fontsize=18)
                plt.ylabel(r'$ 1 H s(x,t) - 1 H s(x,0) $',fontsize=16)
                plt.plot(time[start_idx:final_idx],self.cons_obj[i,start_idx:final_idx]-norm) 
                if logscale: 
                    plt.yscale('symlog',linthresh=1e-14)
                    ax = plt.gca()
                    ymin, ymax = ax.get_ylim()
                    positive_ticks = [0] + [10**exp for exp in range(-12, int(np.log10(ymax)) + 1, 4)]
                    negative_ticks = [-10**exp for exp in range(-12, int(np.log10(-ymin)) + 1, 4)]
                    custom_ticks = negative_ticks[::-1] + positive_ticks
                    ax.set_yticks(custom_ticks)
    
            elif cons_obj_name_i == 'conservation':
                if plot_change:
                    plt.title(r'Change in Conservation',fontsize=18)
                    plt.ylabel(r'$\vert \vert u(x,t) \vert \vert_H$ - $\vert \vert u_0(x) \vert \vert_H$',fontsize=16)
                    plt.ticklabel_format(axis='y',style='sci',scilimits=(0,1))
                    plt.plot(time[start_idx:final_idx],self.cons_obj[i,start_idx:final_idx]-norm)   
                else:
                    plt.title(r'Conservation',fontsize=18)
                    plt.ylabel(r'$\vert \vert u(x,t) \vert \vert_H$',fontsize=16)
                    plt.ticklabel_format(axis='y',style='sci',scilimits=(0,1))
                    plt.plot(time[start_idx:final_idx],self.cons_obj[i,start_idx:final_idx])           
            
            elif cons_obj_name_i == 'a_energy':
                plt.title(r'Change in A-norm Energy',fontsize=18)
                plt.ylabel(r'$\vert \vert u(x,t)^2 \vert \vert_{AH}$ - $\vert \vert u_0(x)^2 \vert \vert_{AH}$',fontsize=16)
                plt.plot(time[start_idx:final_idx],self.cons_obj[i,start_idx:final_idx]-norm) 

            elif cons_obj_name_i == 'max_eig':
                plt.title(r'Maximum Real Eigenvalue of LHS',fontsize=18)
                if normalize_eigs:
                    plt.ylabel(r'$\max \ \Re(\lambda) \Delta x$',fontsize=16)
                else:
                    plt.ylabel(r'$\max \ \Re(\lambda)$',fontsize=16)
                plt.plot(time[start_idx:final_idx],self.cons_obj[i,start_idx:final_idx]*eig_norm) 
                if logscale: plt.yscale('log') 

            elif cons_obj_name_i == 'spec_rad':
                plt.title(r'Spectral Radius of LHS',fontsize=18)
                if normalize_eigs:
                    plt.ylabel(r'$\rho \Delta x = \max \ \vert \lambda \vert \Delta x$',fontsize=16)
                else:
                    plt.ylabel(r'$\rho = \max \ \vert \lambda \vert$',fontsize=16)
                plt.plot(time[start_idx:final_idx],self.cons_obj[i,start_idx:final_idx]*eig_norm) 
                if logscale: plt.yscale('log')

            elif cons_obj_name_i == 'sol_error' or cons_obj_name_i == 'error':
                plt.title(r'Solution Error',fontsize=18)
                plt.ylabel(r'$\vert \vert u - u_e \vert \vert_H$',fontsize=16)
                plt.plot(time[start_idx:final_idx],self.cons_obj[i,start_idx:final_idx])  
                if logscale: plt.yscale('log') 
                
            else:
                print('WARNING: No default plotting set up for '+cons_obj_name_i)
                plt.title(cons_obj_name_i,fontsize=18)
                plt.plot(time[start_idx:final_idx],self.cons_obj[i,start_idx:final_idx]) 
                if logscale: plt.yscale('log') 
                
            if savefile is not None:
                plt.savefig(savefile+'_'+cons_obj_name_i+'.jpg',dpi=600)
            plt.show()

    def force_steady_solution(self, q0_type=None):
        '''
        Modifies the dqdt function to force a steady manufactured solution 
        such that the forcing term is always the negative of the RHS residual
        of the initial condition. If the initial condition is then perturbed,
        this allows for analysis of linearization states. Otherwise, nothing
        should happen and the solution should be steady.
        
        Parameters
        ----------
        q0_type : str, optional
            Indiactes the type of initial solution.
            The default is None, in which case the default q0_type for the
            DiffEq is used.
        '''

        q0 = self.diffeq.set_q0(q0_type=q0_type)
        
        if self.disc_type == 'fd':
            rhs = self.diffeq.dqdt(q0,0.)
            self.dqdt = lambda q, t: self.diffeq.dqdt(q,t) - rhs
        elif self.disc_type == 'div' or self.disc_type == 'had':
            rhs = self.dqdt_1d_div(q0,0.) #TODO: Fix for had and 2D and 3D
            self.dqdt = lambda q, t: self.dqdt_1d_div(q,t) - rhs           
        elif self.disc_type == 'dg':
            if self.weak_form:
                rhs = self.dg_dqdt_weak(q0)
                self.dqdt = lambda q: self.dg_dqdt_weak(q) - rhs  
            else:
                rhs = self.dg_dqdt_strong(q0)
                self.dqdt = lambda q: self.dg_dqdt_strong(q) - rhs  
        else:
            raise Exception('Invalid discretization type')
        
        self.diffeq.exact_sol = lambda *args: q0
        self.diffeq.has_exa_sol = True
        
    def perturb_q0(self, eigmode=True, randnoise=False, ampli=0.001):
        '''
        Modify self.diffeq.set_q0() to add a pertubation according to largest
        real eigenvalue of the approximated spatial operator. This is what is
        in Gassner et. al. 2020 (Stability issues of entropy-stable...). It
        also can add a random noise pertubation.

        Parameters
        ----------
        eigmode : bool, optional
            Whether to use pertubation of largest eigenmode. The default is True.
        randnoise : bool, optional
            Whether to use random noise pertubation. The default is False.
        ampli : float, optional
            Max amplitude of pertubation. The default is 0.001.
        '''
        q0 = self.diffeq.set_q0() # save initial condition
        pert = np.zeros(q0.shape) # initialize pertubation
        if eigmode and randnoise: # if we use both, adjust amplitude
            ampli = ampli/2
        
        # This part adds a pertubation based on the largest real eigenmode
        if eigmode:
            A = self.calc_LHS(q=q0)
            eigvals, eigvecs = np.linalg.eig(A)
            idx = np.argmax(eigvals.real)
            eigmode = eigvecs[:,idx]
            #np.savetxt('eigenmode.csv', eigmode.real, delimiter=',')
            #eigenmode = np.loadtxt('eigmode.csv', delimiter=',')        
            pert += ampli*np.reshape(eigmode.real,q0.shape,'F')/np.max(np.abs(eigmode.real))
        
        # This part adds a random noise pertubation '''
        if randnoise:
            pert += ampli*2*(np.random.rand(*q0.shape)-0.5)
        
        # Now overwrite initial condition function
        self.diffeq.set_q0 = lambda *args: q0 + pert
        
    def plot_sol(self, q=None, **kwargs):
        ''' simply calls the plotting function from diffeq for final sol '''
        if q is None:
            if self.q_sol is None:
                q = self.diffeq.set_q0()
            else:
                if self.q_sol.ndim == 2: q = self.q_sol
                elif self.q_sol.ndim == 3: q = self.q_sol[:,:,-1]
        if 'time' not in kwargs: kwargs['time']=self.t_final
        if 'plot_exa' not in kwargs: kwargs['plot_exa']=True
        if self.disc_nodes in ['lgl','lg','nc']:
            if self.dim==1:
                x50 = np.linspace(0,1,50)
                V = MakeDgOp.VandermondeLagrange1D(x50,self.sbp.x)
                x = V @ self.mesh.x_elem # strictly not correct if nonpolynommial grid warping, but good enough
                from Source.Methods.Functions import kron_neq_lm
                q = kron_neq_lm(V,self.neq_node) @ q
                self.diffeq.plot_sol(q, x, **kwargs)
            elif self.dim==2:
                x50 = np.linspace(0,1,50)
                V = MakeDgOp.VandermondeLagrange1D(x50,self.sbp.x)
                V = np.kron(V,V)
                xy = np.zeros((int(len(x50)**2),2,self.nelem[0]*self.nelem[1]))
                xy[:,0,:] = V @ self.mesh.xy_elem[:,0,:]
                xy[:,1,:] = V @ self.mesh.xy_elem[:,1,:]
                from Source.Methods.Functions import kron_neq_lm
                q = kron_neq_lm(V,self.neq_node) @ q
                self.diffeq.plot_sol(q, xy, **kwargs)
        else:
            self.diffeq.plot_sol(q, **kwargs)

    def plot_sol_error(self, q=None, x=None, idx=-1, time=None, savefile=None, display_time=False,
                   title=None, logscale=True, linear_thresh=None, max_thresh=None, show_fig=True,
                    color='tab:green', linestyle=(0, (1, 1.5)), linewidth=2.3, **kwargs):
        ''' plot the error from a single time as a function of x '''
        assert self.dim == 1, 'ERROR: Only implemented for 1D problems'
        assert self.neq_node == 1, 'ERROR: Only implemented for neq_node=1'
        xlabel = r'$x$'
        ylabel = r'Solution Error $\bm{u} - \bm{u}_{\mathrm{ex}}$'

        if time is None:
            time = self.t_final

        if x is None:
            x = self.mesh.x_elem

        if q is None:
            if self.q_sol.ndim == 2:
                q = self.q_sol
            elif self.q_sol.ndim == 3:
                q = self.q_sol[:,:,idx]

        if self.disc_nodes in ['lg', 'lgl', 'nc']:
            q, x = self.interpolate(u_old=q,return_mesh=True,num_nodes=20)
        er = q - self.diffeq.exact_sol(x=x, time=time)

        plt.figure(figsize=(6,4.4))
        if title is not None: plt.title(title,fontsize=18)
        plt.ylabel(ylabel,fontsize=16)
        plt.xlabel(xlabel,fontsize=16)
        ax = plt.gca()
        if linear_thresh is None:
            linear_thresh = np.min(abs(er))
        if logscale: 
            plt.yscale('symlog',linthresh=linear_thresh)
        else:
            plt.yscale('linear')
            from matplotlib.ticker import FuncFormatter
            def custom_sci_notation(x, pos):
                if x == 0:
                    return "0"
                exponent = int(np.floor(np.log10(abs(x))))
                coeff = x / 10**exponent
                if abs(coeff - 1) < 0.09 :
                    return f"$10^{{{exponent:+d}}}$"
                elif abs(coeff + 1) < 0.09:
                    return f"$-10^{{{exponent:+d}}}$"
                else:
                    return f"${coeff:.1f}\\times 10^{{{exponent:+d}}}$"
            ax.yaxis.set_major_formatter(FuncFormatter(custom_sci_notation))
        plt.grid(which='major',axis='y',linestyle='--',color='gray',linewidth='1')
        ax.tick_params(axis='both', labelsize=13) 
        if max_thresh is not None:
            plt.ylim(-max_thresh,max_thresh)

        plt.plot(x, er[:,0], color=color, linestyle=linestyle, linewidth=linewidth) 
        for elem in range(1,self.nelem):
            plt.plot(x[:,elem], er[:,elem], color=color, linestyle=linestyle, linewidth=linewidth) 

        if display_time and (time is not None):
            props = dict(boxstyle='round', facecolor='white', alpha=0.5)
            ax.text(0.05, 0.95, f'$t={round(time,2)}$', transform=ax.transAxes, 
                    fontsize=15, verticalalignment='top', bbox=props)

        plt.tight_layout()
        if savefile is not None: plt.savefig(savefile, dpi=300)
        if show_fig:
            plt.show()
        plt.close()
        
    def plot_error(self, method=None, savefile=None, extra_fn=None, extra_label=None, title=None):
        ''' plot the error from all time steps as a function of time'''
        errors = self.calc_error(method=method, use_all_t=True)
        steps = np.shape(self.q_sol)[2]
        times = np.linspace(0,self.t_final,steps)
        plt.figure(figsize=(6,4))
        plt.ylabel(r"{0} Error".format(method),fontsize=16)
        plt.xlabel(r"Time",fontsize=16)
        plt.plot(times,errors,label='Error')
        if extra_fn is not None:
            plt.plot(times,extra_fn(times),label=extra_label)
        plt.yscale('log')
        if extra_label is not None:
            plt.legend(loc='best',fontsize=14)
        if title is not None:
            plt.title(title,fontsize=18)
        if savefile is not None:
            plt.savefig(savefile+'.eps', format='eps')
    
    def free_stream(self, print_result = True):
        ''' check free stream preservation '''
        result = np.max(abs(self.dqdt(np.ones(self.qshape),0.)))
        if print_result:
            print('Free Stream Preservation holds to a maximum of {0:.5g}'.format(result))
        else:
            return result
    
    def check_conservation(self, q=None, print_result=True):
        ''' check conservation '''
        if q is None:
            q = self.diffeq.set_q0()
        dqdt = self.dqdt(q,0.)
        result = self.conservation_der(dqdt)
        if print_result:
            print('Derivative of conservation is {0:.5g}'.format(result))
        else:
            return result
        
    def check_stability(self, q=None, print_result=True):
        ''' check stability '''
        if q is None:
            q = self.diffeq.set_q0()
        dqdt = self.dqdt(q,0.)
        result = self.energy_der(q,dqdt)
        if print_result:
            print('Derivative of energy stability is {0:.5g}'.format(result))
        else:
            return result
    
    def calc_cons_obj_rate(self, q=None, t=None, cons_obj_name=None, order=2):
        ''' calculate the rate of change (in time) of the conservation objectives '''
        if q is None:
            q = self.q_sol
        assert q.ndim == 3, 'ERROR: Need solutions in time to calculate rate of change of conservation objectives.' 
        
        if t is None:
            approx_time = False
            if self.cons_obj_name is None:
                approx_time = True
            elif 'time' in self.cons_obj_name:
                idx = np.where('time' == np.array([name.lower() for name in self.cons_obj_name]))[0][0]
                t = self.cons_obj[idx]
            else:
                approx_time = True
            if approx_time:
                print(f'WARNING: No time data in cons_obj_name. Using estimation based on tf={self.t_final}.')
                t = np.linspace(0,self.t_final,np.shape(q)[2])
        
        if cons_obj_name is None:
            cons_obj_name = self.cons_obj_name
            cons_obj = self.cons_obj 
            n_cons_obj = len(cons_obj_name)   
        else:
            n_cons_obj = len(cons_obj_name)
            cons_obj = np.zeros((n_cons_obj,q.shape[2]))

            if self.cons_obj_name is None:
                if any('der' in name.lower() for name in cons_obj_name):
                    raise Exception('ERROR: Cannot calculate conservation objectives with derivatives since dqdt was not stored.')
                dqdt = 0.
                cons_obj = np.zeros((n_cons_obj,q.shape[2]))
            else:
                # note: some cons_obj may have already been calculated
                remaining_cons_obj_name = [name for name in cons_obj_name if name not in self.cons_obj_name]
                if any('der' in name.lower() for name in remaining_cons_obj_name):
                    raise Exception('ERROR: Cannot calculate conservation objectives with derivatives since dqdt was not stored.')
                dqdt = 0.
                if len(remaining_cons_obj_name) == n_cons_obj:
                    cons_obj = self.calc_cons_obj(q, t, dqdt, cons_obj_name)
                elif len(remaining_cons_obj_name) == 0:
                    cons_obj = self.cons_obj
                else:
                    remaining_cons_obj = self.calc_cons_obj(q, t, dqdt, remaining_cons_obj_name)
                    cons_obj = np.zeros((n_cons_obj,q.shape[2]))
                    for i, name in enumerate(cons_obj_name):
                        if name in self.cons_obj_name:
                            idx = self.cons_obj_name.index(name)
                            cons_obj[i, :] = self.cons_obj[idx, :]
                        else:
                            idx = remaining_cons_obj_name.index(name)
                            cons_obj[i, :] = remaining_cons_obj[idx, :]

        cons_obj_rate = np.zeros((n_cons_obj,q.shape[2]))
        for i in range(n_cons_obj):
            if order == 2:
                # second order central difference (+ forward/backward at ends)
                cons_obj_rate[i,0] = (cons_obj[i,1]-cons_obj[i,0])/(t[1]-t[0])
                cons_obj_rate[i,-1] = (cons_obj[i,-1]-cons_obj[i,-2])/(t[-1]-t[-2])
                cons_obj_rate[i,1:-1] = (cons_obj[i,2:]-cons_obj[i,:-2])/(t[2:]-t[:-2])
            elif order == 4:
                # fourth order central difference (+ forward/backward at ends)
                cons_obj_rate[i,0] = (-25*cons_obj[i,0]+48*cons_obj[i,1]-36*cons_obj[i,2]+16*cons_obj[i,3]-3*cons_obj[i,4])/(12*(t[1]-t[0]))
                cons_obj_rate[i,1] = (-3*cons_obj[i,0]-10*cons_obj[i,1]+18*cons_obj[i,2]-6*cons_obj[i,3]+cons_obj[i,4])/(12*(t[2]-t[1]))
                cons_obj_rate[i,-1] = (25*cons_obj[i,-1]-48*cons_obj[i,-2]+36*cons_obj[i,-3]-16*cons_obj[i,-4]+3*cons_obj[i,-5])/(12*(t[-1]-t[-2]))
                cons_obj_rate[i,-2] = (3*cons_obj[i,-1]+10*cons_obj[i,-2]-18*cons_obj[i,-3]+6*cons_obj[i,-4]-cons_obj[i,-5])/(12*(t[-2]-t[-3]))
                cons_obj_rate[i,2:-2] = (-cons_obj[i,0:-4]+8*cons_obj[i,1:-3]-8*cons_obj[i,3:-1]+cons_obj[i,4:])/(12*(t[2:-2]-t[1:-3]))
        return cons_obj_rate
            
    def interpolate(self, u_old=None, x_new=None, x_old=None, 
                    return_mesh=False, num_nodes=11):
        """
        Interpolates the values from x_old to x_new using barycentric lagrange interpolation.
        
        Parameters:
        x_old (array): The original x values (reference element).
        u_old (array): The original function values at x_old.
        x_new (array): The new x values to interpolate to (reference element).
        
        Returns:
        u_new (array): The interpolated function values at x_new.
        """
        assert self.disc_nodes in ['lgl','lg','nc'], 'ERROR: Interpolation only implemented for element-type.'

        from Source.Disc.MakeDgOp import MakeDgOp
        from Source.Methods.Functions import lm_gv, kron_neq_lm

        if u_old is None:
            if self.q_sol is None:
                raise Exception('ERROR: No solution to interpolate.')
            elif self.q_sol.ndim == 2:
                u_old = self.q_sol
            elif self.q_sol.ndim == 3:
                u_old = self.q_sol[:,:,-1]
            else:
                raise Exception('ERROR: q_sol must be 2D or 3D array.')

        if x_new is None:
            x_new = np.linspace(0,1,num_nodes,endpoint=True)

        if x_old is None:
            x_old = self.mesh.x_op

        assert x_old.ndim == 1 and len(x_old) == self.nen, 'ERROR: x_old must be 1-dimensional and of length self.nen.'
        x_old = np.reshape(x_old, (self.nen,1))
        
        # Build the Vandermonde matrix for x_new
        V = MakeDgOp.VandermondeLagrange1D(x_new, x_old)
        Vsys = kron_neq_lm(V,self.neq_node)

        if u_old.ndim == 2:
            u_new = lm_gv(Vsys, u_old)
            if return_mesh:
                new_mesh = lm_gv(V, self.mesh.x_elem)
        else:
            u_new = Vsys @ u_old
            if return_mesh:
                new_mesh = V @ self.mesh.x

        if return_mesh:
            return u_new, new_mesh
        else:
            return u_new
        
    def dispersion_analysis(self, plot=True, num_k=10, all_modes=True,
                            return_omegas=False):
        ''' Perform a dispersion analysis following
          COMPARISON OF GAUSS AND GAUSS–LOBATTO INTEGRATION, Gassner & Kopriva 2011  '''
        assert self.diffeq.diffeq_name == 'LinearConvection'
        assert self.dim == 1
        assert self.bc == 'periodic'
        assert self.mesh.warp_factor == 0.0
        assert self.sparse == False
        assert self.sat.sparse == False
        #a = self.diffeq.para[0]
        dx = (self.mesh.xmax - self.mesh.xmin)/self.nelem
        Hinv = np.linalg.inv(self.sbp.H)
        tLtLT = np.complex128(Hinv @ self.sbp.ta @ self.sbp.ta.T)
        tLtRT = np.complex128(Hinv @ self.sbp.ta @ self.sbp.tb.T)
        D = np.complex128(self.sbp.D)
        diss = np.complex128(Hinv @ self.adiss.dispersion_analysis())

        ks = np.linspace(0, self.nen*np.pi/dx, num_k, endpoint=True)
        Ks = np.linspace(0, np.pi, num_k, endpoint=True)
        if all_modes:
            Omegas = np.zeros((self.nen,num_k),dtype=complex)
        else:
            Omegas = np.zeros(num_k,dtype=complex)

        for i, k in enumerate(ks):
            K = k*dx
            if self.sat.diss_type == 'lf':
                A = (D + tLtLT - np.exp(-1j*K)*tLtRT + diss)/1j
            else:
                raise Exception('dispersion analysis only implemented for LF Sats')
            
            eigvals = np.linalg.eigvals(A)
            if all_modes:
                # organize the modes in the same order each time
                if k==0: 
                    reorder = eigvals
                else:
                    reorder = np.zeros(self.nen,dtype=complex)
                    if self.nen < 10:
                        for j in range(self.nen):
                            reorder[j] = min(eigvals, key=lambda z: abs(z-Omegas[j,i-1]))
                    else: reorder = eigvals
                Omegas[:,i] = reorder
            else:
                # select only physical mode
                if i == 0:
                    Omegas[i] = min(eigvals, key=lambda z: abs(z))
                elif i < 1:
                    # can't match with derivative yet
                    Omegas[i] = min(eigvals, key=lambda z: abs(z-Omegas[i-1]))
                elif i < 3:
                    Omegas[i] = min(eigvals, key=lambda z: (
                        0.5 * abs(z-Omegas[i-1]) +
                        0.5 * abs(z.real - 2*Omegas[i-1].real + Omegas[i - 2].real) +
                        0.5 * abs(z.imag - 2*Omegas[i-1].imag + Omegas[i - 2].imag) ))
                else:
                    Omegas[i] = min(eigvals, key=lambda z: (
                        0.5 * abs(z-Omegas[i-1]) +
                        0.5 * abs(2*z.real - 5*Omegas[i-1].real + 4*Omegas[i - 2].real - Omegas[i-3].real) +
                        0.5 * abs(2*z.imag - 5*Omegas[i-1].imag + 4*Omegas[i - 2].imag- Omegas[i-3].imag) ))

        Omegas = Omegas/self.nen

        if plot:
            plt.figure()
            plt.plot(Ks,Ks,'--',color='black')
            if all_modes:
                for i in range(self.nen):
                    if self.nen < 10:
                        plt.plot(Ks,np.real(Omegas[i,:]))
                    else:
                        plt.plot(Ks,np.real(Omegas[i,:]),linestyle='',
                                 marker='.',color='tab:blue',markersize=1)
            else:
                plt.plot(Ks,np.real(Omegas))
            plt.xlabel(r'$K$',fontsize=16)
            plt.ylabel(r'$\Re(\Omega)$',fontsize=16)
            plt.title('Dispersion Relation',fontsize=18)
            plt.show()

            plt.figure()
            plt.plot(Ks,np.zeros_like(Ks),'--',color='black')
            if all_modes:
                for i in range(self.nen):
                    if self.nen < 10:
                        plt.plot(Ks,np.imag(Omegas[i,:]))
                    else:
                        plt.plot(Ks,np.imag(Omegas[i,:]),linestyle='',
                                 marker='.',color='tab:blue',markersize=1)
            else:
                plt.plot(Ks,np.imag(Omegas))
            plt.xlabel(r'$K$',fontsize=16)
            plt.ylabel(r'$\Im(\Omega)$',fontsize=16)
            plt.title('Dissipation Relation',fontsize=18)
            plt.show()

        if return_omegas:
            return Omegas